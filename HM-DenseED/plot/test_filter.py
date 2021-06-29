#Reference: https://github.com/zabaras/pde-surrogate/blob/master/utils/image_gradient.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import torch
import torch.autograd as ag
import math
##=========================
def gaussian_filter1d_weights(sigma, order=0, truncate=4.0):
    """One-dimensional Gaussian filter.
    https://github.com/scipy/scipy/blob/v0.16.1/scipy/ndimage/filters.py#L181

    Parameters
    ----------
    %(input)s
    sigma : scalar
        standard deviation for Gaussian kernel
    %(axis)s
    order : {0, 1, 2, 3}, optional
        An order of 0 corresponds to convolution with a Gaussian
        kernel. An order of 1, 2, or 3 corresponds to convolution with
        the first, second or third derivatives of a Gaussian. Higher
        order derivatives are not implemented
    %(output)s
    %(mode)s
    %(cval)s
    truncate : float, optional
        Truncate the filter at this many standard deviations.
        Default is 4.0.
    Returns
    -------
    gaussian_filter1d : ndarray
    """
    if order not in range(4):
        raise ValueError('Order outside 0..3 not implemented')
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd = sd * sd
    # calculate the kernel:
    for ii in range(1, lw + 1):
        tmp = math.exp(-0.5 * float(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    # implement first, second and third order derivatives:
    if order == 1:  # first derivative
        weights[lw] = 0.0
        for ii in range(1, lw + 1):
            x = float(ii)
            tmp = -x / sd * weights[lw + ii]
            weights[lw + ii] = -tmp
            weights[lw - ii] = tmp
    elif order == 2:  # second derivative
        weights[lw] *= -1.0 / sd
        for ii in range(1, lw + 1):
            x = float(ii)
            tmp = (x * x / sd - 1.0) * weights[lw + ii] / sd
            weights[lw + ii] = tmp
            weights[lw - ii] = tmp
    elif order == 3:  # third derivative
        weights[lw] = 0.0
        sd2 = sd * sd
        for ii in range(1, lw + 1):
            x = float(ii)
            tmp = (3.0 - x * x / sd) * x * weights[lw + ii] / sd2
            weights[lw + ii] = -tmp
            weights[lw - ii] = tmp

    return np.array(weights)
    

class GaussianFilter(object):
    """Gaussian smoothing

    Only use `reflect` mode for padding
    """
    def __init__(self, sigma=1.0, truncate=4.0, order=0, device='cpu'):

        gaussian_weights_1d = gaussian_filter1d_weights(sigma, 
            order=order, truncate=truncate)
        weights = np.expand_dims(gaussian_weights_1d, 1)

        self.weights = torch.FloatTensor(
            np.matmul(weights, weights.T)).unsqueeze(0).unsqueeze(0).to(device)

    def __call__(self, image):
        # image: (B, C, H, W)
        padding = (self.weights.shape[-1] - 1) // 2
        image = F.pad(image, _quadruple(padding), mode='reflect')
        channels = image.shape[1]
        weights = self.weights.repeat(channels, 1, 1, 1)
        return F.conv2d(image, weights, bias=None, stride=1, padding=0, groups=channels)


