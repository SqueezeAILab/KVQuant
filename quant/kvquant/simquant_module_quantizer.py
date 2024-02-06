import numpy as np
import torch
import torch.nn as nn
import math
from sklearn.cluster import KMeans

import torch
from torch.distributions import Normal

def round_to_nearest_pole_sim(w, poles):
    """
    w: weight/act values (1d vector)
    poles: tuple of values

    Round the numbers in w to the nearest value in poles.
    """
    stack = []
    for c in poles:
        diff = (w - c).abs()
        stack.append(diff)
    diff = torch.stack(stack)
    idx = diff.argmin(axis=0)
    aug = 0
    freq = []
    for i, c in enumerate(poles):
        aug += (idx == i) * c

    return aug

def get_outliers(
    w,
    channel=-1,
    outlier_threshold_upper=-1,
    outlier_threshold_lower=-1
):
    """
    w: weight/act values (1d vector)
    channel: which dimension to share scaling factors along
    outlier_threshold_upper: upper outlier thresholds
    outlier_threshold_lower: lower outlier thresholds

    Detect outliers above upper threshold / below lower threshold
    """
    # only use either per-channel or per-token outlier
    outlier_threshold_upper = outlier_threshold_upper.unsqueeze(channel)
    outlier_threshold_lower = outlier_threshold_lower.unsqueeze(channel)

    under_lower = w < outlier_threshold_lower
    above_upper = w > outlier_threshold_upper

    outlier_mask = torch.logical_or(under_lower, above_upper)
    return outlier_mask

def get_outliers_dynamic(
    w,
    channel=-1,
    thresh=0.999
):
    """
    w: weight/act values (1d vector)
    channel: which dimension to share scaling factors along
    thresh: percentile for outlier threshold computation

    Detect outliers above upper threshold / below lower threshold
    """

    t = 1-((1-thresh)/2)
    w = w.float()

    # only use either per-channel or per-token outlier
    outlier_threshold_upper = torch.quantile(w, t, dim=channel)
    outlier_threshold_lower = torch.quantile(w, 1-t, dim=channel)

    outlier_threshold_upper = outlier_threshold_upper.unsqueeze(channel)
    outlier_threshold_lower = outlier_threshold_lower.unsqueeze(channel)

    under_lower = w <= outlier_threshold_lower
    above_upper = w >= outlier_threshold_upper

    outlier_mask = torch.logical_or(under_lower, above_upper)
    return outlier_mask

# integer quantization function
def quant_fn_zp(
    inp,
    bits=8,
    qchannel = -1,
    dynamicquantization=False,
    include_sparse=False,
    outlier_mask=None,
    maxval=-1,
    minval=-1,
):
    """
    inp: weight/act values (2d matrix)
    bits: number of bits for quantization
    qchannel: which dimension to share scaling factors along
    dynamicquantization: whether to compute scaling factors / outlier thresholds online
    include_sparse: whether to use dense-and-sparse quantization
    outlier_mask: positions of outlier values
    maxval: upper outlier thresholds (if not dynamically computed)
    minval: lower outlier thresholds (if not dynamically computed)

    Performs simulated integer quantization
    """

    # set quantization threshold dynamically
    if dynamicquantization:
        if include_sparse:
            outliers = inp * outlier_mask
            median = torch.median(inp, dim=qchannel).values
            median = median.unsqueeze(qchannel)
            median_mask = median * outlier_mask

            # recenter using median to avoid having outliers skew quant distribution
            tmp_inp = inp - outliers + median_mask
            maxval = torch.max(tmp_inp, dim=qchannel).values
            minval = torch.min(tmp_inp, dim=qchannel).values
        else:
            maxval = torch.max(inp, dim=qchannel).values
            minval = torch.min(inp, dim=qchannel).values

    # compute offset here:
    rangeval = (maxval - minval)
    qx = (2**bits - 1) / rangeval

    # set offset
    offset = torch.round(minval * qx)
    offset = offset.clamp(-(2**bits - 1), 0)

    offset = offset.unsqueeze(qchannel)
    qx = qx.unsqueeze(qchannel)

    # need to handle outlier removal
    if include_sparse:
        outliers = inp * outlier_mask
        inp = inp - outliers

    # scale and subtract offset
    qinp = torch.round(qx * inp - offset)

    #clipping (just for debugging purposes)
    qinp = torch.clip(qinp, min=0, max=2**bits - 1)

    #rescale
    qinp_out = (qinp + offset) / qx

    # add outliers back
    if include_sparse:
        qinp_out[outlier_mask] = 0
        qinp_out = qinp_out + outliers

    qinp_out = torch.nan_to_num(qinp_out, nan=0.0, posinf=0.0, neginf=0.0)
    return qinp_out

def quant_fn_nf(
    inp,
    bits=8,
    qchannel = -1,
    dynamicquantization=False,
    include_sparse=False,
    outlier_mask=None,
    maxval=-1,
    minval=-1,
    nf_lut=None
):
    """
    inp: weight/act values (2d matrix)
    bits: number of bits for quantization
    qchannel: which dimension to share scaling factors along
    dynamicquantization: whether to compute scaling factors / outlier thresholds online
    include_sparse: whether to use dense-and-sparse quantization
    outlier_mask: positions of outlier values
    maxval: upper outlier thresholds (if not dynamically computed)
    minval: lower outlier thresholds (if not dynamically computed)
    nf_lut: NormalFloat signpost values

    Performs simulated NormalFloat quantization
    """

    # set quantization threshold dynamically
    if dynamicquantization:
        if include_sparse:
            outliers = inp * outlier_mask
            median = torch.median(inp, dim=qchannel).values
            median = median.unsqueeze(qchannel)
            median_mask = median * outlier_mask

            # recenter using mean to avoid having outliers skew quant distribution
            tmp_inp = inp - outliers + median_mask
            maxval = torch.max(tmp_inp, dim=qchannel).values
            minval = torch.min(tmp_inp, dim=qchannel).values
        else:
            maxval = torch.max(inp, dim=qchannel).values
            minval = torch.min(inp, dim=qchannel).values

    # compute offset here:
    offset = (maxval + minval) / 2
    rangeval = (maxval - minval) / 2
    offset = offset.unsqueeze(qchannel)
    rangeval = rangeval.unsqueeze(qchannel)

    # subtract offset
    inp = inp - offset

    # need to handle outlier removal here due to issues with zeroing out non-outliers
    if include_sparse:
        outliers = inp * outlier_mask
        inp = inp - outliers

    #dividing by range to normalize to [-1,1]
    inp_scaled = inp / rangeval

    Q = round_to_nearest_pole_sim(inp_scaled.flatten(), nf_lut)
    qinp_out = Q.reshape(inp.shape).half().cuda()
    qinp_out = qinp_out * rangeval

    # add outliers back
    if include_sparse:
        qinp_out = qinp_out + outliers

    #shift by offset
    qinp_out = qinp_out + offset
    qinp_out = torch.nan_to_num(qinp_out, nan=0.0, posinf=0.0, neginf=0.0) #TODO: debug (shouldn't be necessary)

    return qinp_out

def quant_fn_nuq_recon(
    inp,
    bits=8,
    qchannel = -1,
    dynamicquantization=False,
    include_sparse=False,
    outlier_mask=None,
    maxval=-1,
    minval=-1,
    lut=None,
    norm=False,
    normscale=None,
    normoffset=None
):
    """
    inp: weight/act values (2d matrix)
    bits: number of bits for quantization
    qchannel: which dimension to share scaling factors along
    dynamicquantization: whether to compute scaling factors / outlier thresholds online
    include_sparse: whether to use dense-and-sparse quantization
    outlier_mask: positions of outlier values
    maxval: upper outlier thresholds (if not dynamically computed)
    minval: lower outlier thresholds (if not dynamically computed)
    lut: NUQ signpost values
    norm: whether to use Q-Norm
    normscale: scaling for Q-Norm
    normoffset: shift for Q-Norm

    Performs simulated NUQ quantization
    """

    # set quantization threshold dynamically
    if dynamicquantization:
        if include_sparse:
            outliers = inp * outlier_mask
            median = torch.median(inp, dim=qchannel).values
            median = median.unsqueeze(qchannel)
            median_mask = median * outlier_mask

            # recenter using mean to avoid having outliers skew quant distribution
            tmp_inp = inp - outliers + median_mask
            maxval = torch.max(tmp_inp, dim=qchannel).values
            minval = torch.min(tmp_inp, dim=qchannel).values
        else:
            maxval = torch.max(inp, dim=qchannel).values
            minval = torch.min(inp, dim=qchannel).values

    # compute offset here:
    offset = (maxval + minval) / 2
    rangeval = (maxval - minval) / 2
    offset = offset.unsqueeze(qchannel)
    rangeval = rangeval.unsqueeze(qchannel)

    # subtract offset
    inp = inp - offset

    # need to handle outlier removal here due to issues with zeroing out non-outliers
    if include_sparse:
        outliers = inp * outlier_mask
        inp = inp - outliers

    #dividing by range to normalize to [-1,1]
    inp_scaled = inp / rangeval

    # round to nearest LUT entry
    lut_cuda = torch.tensor(lut[0]).to(inp_scaled.device)
    Q = round_to_nearest_pole_sim(inp_scaled.flatten(), lut_cuda)
    qinp_out = Q.reshape(inp.shape).float().to(inp_scaled.device)

    if norm:
        normscale = normscale.to(inp_scaled.device)
        normoffset = normoffset.to(inp_scaled.device)
        qinp_out = qinp_out*normscale + normoffset

    # un-normalize
    qinp_out = qinp_out * rangeval

    # add outliers back
    if include_sparse:
        qinp_out[outlier_mask] = 0
        qinp_out = qinp_out + outliers

    #shift by offset
    qinp_out = qinp_out + offset
    qinp_out = torch.nan_to_num(qinp_out, nan=0.0, posinf=0.0, neginf=0.0) #TODO: debug (shouldn't be necessary)

    return qinp_out.float()

# simquant quantizer (calibration)
class SimQuant:
    def __init__(
                    self,
                    layer,
                    bits,
                    perchannel=True,
                    qchannel=0,
                    include_rope=False
                ):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.perchannel = perchannel
        self.qchannel = qchannel
        self.bits = bits

        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.nsamples = 0

        self.out = None

    def add_batch(self, inp, out):
        if len(out.shape) == 2:
            out = out.unsqueeze(0)
        tmp = out.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(out.shape) == 3:
                out = out.reshape((-1, self.rows))
        self.nsamples += tmp

        if self.out == None:
            self.out = out.clone()
        else:
            self.out = torch.cat((self.out, out.clone()), dim=0)

    def quantize(
        self,
        include_sparse=False,
        sparsity_threshold=0.999,
        nuq=False,
        fisher=False,
        norm=False
    ):

        # for now, just update threshold here
        if include_sparse:
            t = 1-((1-sparsity_threshold)/2)
        else:
            t = 1 #use min-max quantization

        #TODO - if not using sparsity, use a different threshold for min-max quant?
        data = self.out.float().cpu().numpy()
        if self.perchannel:
            #per-channel - remove tokenwise outliers and normalize range to [-1,1]
            outlier_threshold_upper = np.percentile(data, t*100, axis=self.qchannel)
            outlier_threshold_lower = np.percentile(data, (1-t)*100, axis=self.qchannel)
        else:
            #per-token - remove tokenwise outliers and normalize range to [-1,1]
            assert(False) # not currently supported

        # convert to torch
        data = torch.tensor(data)
        outlier_threshold_upper = torch.tensor(outlier_threshold_upper).unsqueeze(self.qchannel)
        outlier_threshold_lower = torch.tensor(outlier_threshold_lower).unsqueeze(self.qchannel)

        # range and offset
        rangeval = (outlier_threshold_upper - outlier_threshold_lower) / 2
        zeropoint = (outlier_threshold_upper + outlier_threshold_lower) / 2

        # shift by offset
        data_shifted = data - zeropoint

        # normalize by rangeval into [-1,1]
        data_shifted_normalized = data_shifted / rangeval

        #get outliers (need to mask out for kmeans)
        outlier_mask = torch.logical_or((data_shifted_normalized > 1), (data_shifted_normalized < -1))

        if nuq:
            centroids = []
            act_distn_np = data_shifted_normalized.flatten()
            n_cluster = 2 ** self.bits

            outlier_mask_unflattened = outlier_mask
            outlier_mask = outlier_mask.flatten()
            act_distn_np_without_outliers = act_distn_np[~outlier_mask]
            act_distn_np_without_outliers = act_distn_np_without_outliers.float().cpu().numpy().reshape(-1, 1)

            # load fisher info
            if fisher is not None:
                fisher_info = fisher.flatten()
                fisher_info_tmp_without_outliers = fisher_info[~outlier_mask]
                kmeans = KMeans(
                    n_clusters=n_cluster,
                    random_state=0,
                    n_init="auto",
                    max_iter=50,
                ).fit(
                    act_distn_np_without_outliers,
                    sample_weight=fisher_info_tmp_without_outliers,
                )
            else:
                kmeans = KMeans(
                    n_clusters=n_cluster,
                    random_state=0,
                    n_init="auto",
                    max_iter=50,
                ).fit(
                    act_distn_np_without_outliers
                )

            centroids.append(kmeans.cluster_centers_)

            #Q-Norm
            if norm:
                centroid = torch.tensor(centroids[0])
                aug = torch.tensor(data_shifted_normalized)
                not_outlier_mask_unflattened = ~outlier_mask_unflattened

                m1 = (aug*not_outlier_mask_unflattened).sum()/not_outlier_mask_unflattened.sum()
                not_outlier_mask_unqueeze = not_outlier_mask_unflattened.sum()
                stdev1 = torch.sqrt(torch.sum(((aug - m1)*not_outlier_mask_unflattened)**2) / not_outlier_mask_unqueeze)

                aug, freq = round_to_nearest_pole_sim(aug, centroid, return_freq=True)

                m2 = (aug*not_outlier_mask_unflattened).sum()/not_outlier_mask_unflattened.sum()
                stdev2 = torch.sqrt(torch.sum(((aug - m2)*not_outlier_mask_unflattened)**2) / not_outlier_mask_unqueeze)

                normscale = (stdev1 / stdev2)
                normoffset = (- m2) * (stdev1 / stdev2) + m1

                return outlier_threshold_upper, outlier_threshold_lower, centroids, normscale, normoffset
            else:
                return outlier_threshold_upper, outlier_threshold_lower, centroids
        else:
            # not using NUQ
            return outlier_threshold_upper, outlier_threshold_lower

    def free(self):
        self.out = None
        self.qout = None
        torch.cuda.empty_cache()

# drop-in layer replacement class
class QuantLinearSim(nn.Module):
    def __init__(
                    self,
                    name,
                    bits,
                    quantizer,
                    infeatures,
                    outfeatures,
                    weight,
                    bias,
                    perchannel=True,
                    include_sparse=False,
                    sparsity_threshold=0.999,
                    dynamicquantization=False,
                    nuq=False,
                    nf_nuq=True,
                    norm=False
                ):

        super().__init__()
        if bits not in [2,3,4,5]:
            raise NotImplementedError("Only 3, 4, 5 bits are supported.")
        self.name = name
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits

        self.weight = weight.T.detach().cpu()
        if bias:
            self.bias = bias.detach().cpu()
        else:
            self.bias = None

        self.perchannel = perchannel
        self.dynamicquantization = dynamicquantization

        if perchannel:
            self.qchannel = 0
        else: #per-token quant
            self.qchannel = -1

        self.ochannel = self.qchannel

        self.include_sparse = include_sparse
        self.sparsity_threshold = sparsity_threshold
        self.outlier_threshold_upper = torch.tensor(quantizer[0]).cuda().flatten().half()
        self.outlier_threshold_lower = torch.tensor(quantizer[1]).cuda().flatten().half()

        self.nuq = nuq
        self.nf_nuq = nf_nuq
        if self.nuq and not self.nf_nuq:
            self.lut = quantizer[2]
        else:
            self.lut = None

        if norm:
            self.normscale = quantizer[3]
            self.normoffset = quantizer[4]
            self.norm = True
        else:
            self.norm = False
            self.normscale = None
            self.normoffset = None

        # for normalfloat support - compute NF signposts
        if self.nf_nuq:
            dist = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
            # get evenly spaced percentile values

            num_signposts_pos = (2 ** (self.bits - 1)) + 1 # for pos half
            num_signposts_neg = (2 ** (self.bits - 1)) # for neg half

            self.nf_signposts_negative = []
            self.nf_signposts_positive = []

            # from https://arxiv.org/pdf/2306.06965.pdf
            offsets = [0.5*(1/32 + 1/30), 1 - 0.5*(1/32 + 1/30)]
            list1 = [offsets[0]]
            spacing = (0.5 - offsets[0]) / (2 ** (self.bits - 1) - 1)

            add = offsets[0]
            for i in range(num_signposts_neg - 1):
                add += spacing
                list1.append(add)

            list2 = []
            spacing = (offsets[1] - 0.5) / (2 ** (self.bits - 1)) #1 extra space
            add = 0.5
            for i in range(num_signposts_pos - 1):
                list2.append(add)
                add += spacing
            list2.append(offsets[-1])

            # first do negative part [0->0.5]
            for i in range(num_signposts_neg):
                v1 = list1[i]
                val = dist.icdf(torch.tensor([v1])).data.numpy()
                self.nf_signposts_negative.append(torch.tensor(val).item())

            # next do positive part [0.5->1]
            for i in range(num_signposts_pos):
                v1 = list2[i]
                val = dist.icdf(torch.tensor([v1])).data.numpy()
                self.nf_signposts_positive.append(torch.tensor(val).item())

            signpost_neg_min = self.nf_signposts_negative[0]
            signpost_neg_max = self.nf_signposts_negative[-1]
            rangeval = abs(signpost_neg_min)-abs(signpost_neg_max)
            off = abs(signpost_neg_max)
            for s in range(len(self.nf_signposts_negative)):
                self.nf_signposts_negative[s] = (self.nf_signposts_negative[s] + off) / rangeval

            signpost_pos_min = self.nf_signposts_positive[0]
            signpost_pos_max = self.nf_signposts_positive[-1]
            rangeval = abs(signpost_pos_max)-abs(signpost_pos_min)
            off = abs(signpost_pos_min)

            for s in range(len(self.nf_signposts_positive)):
                self.nf_signposts_positive[s] = (self.nf_signposts_positive[s] - off) / rangeval

            del self.nf_signposts_positive[0]

            # delete last negative value and merge
            self.nf_signposts = self.nf_signposts_negative + self.nf_signposts_positive

            assert (len(self.nf_signposts) == (2 ** self.bits))

    #replacement forward pass
    def forward(self, x, other_mat=None):

        out_shape = x.shape[:-1] + (self.outfeatures, )
        x = x.reshape(-1,x.shape[-1])

        # copying weight to / from device during evaluation lets us evaluate
        # a large model with limitted memory usage

        self.weight = self.weight.to(x.device)
        if self.bias is not None:
            self.bias = self.bias.to(x.device)

        x = x.half() # for now cast to fp16 and back (quantization code assumes fp32)
        y = x @ self.weight
        y = y + self.bias if self.bias is not None else y
        y = y.float()

        # if using dense-and-sparse quantization, detect outliers in output tensor
        if self.include_sparse:
            if self.dynamicquantization:
                outlier_mask = get_outliers_dynamic(
                    y,
                    channel=self.ochannel,
                    thresh=self.sparsity_threshold
                )
            else:
                self.outlier_threshold_upper = self.outlier_threshold_upper.to(y.device)
                self.outlier_threshold_lower = self.outlier_threshold_lower.to(y.device)
                outlier_mask = get_outliers(
                    y,
                    channel=self.ochannel,
                    outlier_threshold_upper=self.outlier_threshold_upper,
                    outlier_threshold_lower=self.outlier_threshold_lower
                )
        else:
            outlier_mask = None

        # quantize output tensor
        if self.nuq:
            if self.nf_nuq:
                y = quant_fn_nf(
                    y,
                    bits=self.bits,
                    qchannel=self.qchannel,
                    maxval=self.outlier_threshold_upper,
                    minval=self.outlier_threshold_lower,
                    include_sparse=self.include_sparse,
                    outlier_mask=outlier_mask,
                    dynamicquantization=self.dynamicquantization,
                    nf_lut=self.nf_signposts
                )
            else:
                y = quant_fn_nuq_recon(
                    y,
                    bits=self.bits,
                    qchannel=self.qchannel,
                    maxval=self.outlier_threshold_upper,
                    minval=self.outlier_threshold_lower,
                    include_sparse=self.include_sparse,
                    outlier_mask=outlier_mask,
                    dynamicquantization=self.dynamicquantization,
                    lut=self.lut,
                    norm=self.norm,
                    normscale=self.normscale,
                    normoffset=self.normoffset
                )

        else:
            # low-bit uniform simulated quant
            y = quant_fn_zp(
                y,
                bits=self.bits,
                qchannel=self.qchannel,
                maxval=self.outlier_threshold_upper,
                minval=self.outlier_threshold_lower,
                include_sparse=self.include_sparse,
                outlier_mask=outlier_mask,
                dynamicquantization=self.dynamicquantization
            )

        self.weight = self.weight.cpu()
        if self.bias is not None:
            self.bias = self.bias.cpu()

        y = y.reshape(out_shape)

        y = y.half()
        return y

# update modules
def make_quant_sim(
                    module,
                    quantizers,
                    bits,
                    name='',
                    perchannel=True,
                    include_sparse=False,
                    sparsity_threshold=0.999,
                    dynamicquantization=False,
                    nuq=False,
                    nf_nuq=True,
                    norm=False
                  ):
    if isinstance(module, QuantLinearSim):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in quantizers.keys():
            delattr(module, attr)
            setattr(module, attr, QuantLinearSim(
                                                    name1,
                                                    bits,
                                                    quantizers[name1],
                                                    tmp.in_features,
                                                    tmp.out_features,
                                                    tmp.weight,
                                                    tmp.bias is not None,
                                                    perchannel=perchannel,
                                                    include_sparse=include_sparse,
                                                    sparsity_threshold=sparsity_threshold,
                                                    dynamicquantization=dynamicquantization,
                                                    nuq=nuq,
                                                    nf_nuq=nf_nuq,
                                                    norm=norm
                                                ))
        del tmp
    for name1, child in module.named_children():
        make_quant_sim(
                        child,
                        quantizers,
                        bits,
                        name + '.' + name1 if name != '' else name1,
                        perchannel=perchannel,
                        include_sparse=include_sparse,
                        sparsity_threshold=sparsity_threshold,
                        dynamicquantization=dynamicquantization,
                        nuq=nuq,
                        nf_nuq=nf_nuq,
                        norm=norm
                      )
