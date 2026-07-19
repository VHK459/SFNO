#legengre
import numpy as np
import torch
import torch.nn as nn
import torch.fft
import numpy as np
from torch.utils.checkpoint import checkpoint

def clm(l, m):
    """
    defines the normalization factor to orthonormalize the Spherical Harmonics
    """
    return np.sqrt((2*l + 1) / 4 / np.pi) * np.sqrt(np.math.factorial(l-m) / np.math.factorial(l+m))

def legpoly(mmax, lmax, x, norm="ortho", inverse=False, csphase=True):
    r"""
    Computes the values of (-1)^m c^l_m P^l_m(x) at the positions specified by x.
    The resulting tensor has shape (mmax, lmax, len(x)). The Condon-Shortley Phase (-1)^m
    can be turned off optionally.

    method of computation follows
    [1] Schaeffer, N.; Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Rapp, R.H.; A Fortran Program for the Computation of Gravimetric Quantities from High Degree Spherical Harmonic Expansions, Ohio State University Columbus; report; 1982;
        https://apps.dtic.mil/sti/citations/ADA123406
    [3] Schrama, E.; Orbit integration based upon interpolated gravitational gradients
    """

    # compute the tensor P^m_n:
    nmax = max(mmax,lmax)
    vdm = np.zeros((nmax, nmax, len(x)), dtype=np.float64)
        
    norm_factor = 1. if norm == "ortho" else np.sqrt(4 * np.pi)
    norm_factor = 1. / norm_factor if inverse else norm_factor

    # initial values to start the recursion
    vdm[0,0,:] = norm_factor / np.sqrt(4 * np.pi)

    # fill the diagonal and the lower diagonal
    for l in range(1, nmax):
        vdm[l-1, l, :] = np.sqrt(2*l + 1) * x * vdm[l-1, l-1, :]
        vdm[l, l, :] = np.sqrt( (2*l + 1) * (1 + x) * (1 - x) / 2 / l ) * vdm[l-1, l-1, :]

    # fill the remaining values on the upper triangle and multiply b
    for l in range(2, nmax):
        for m in range(0, l-1):
            vdm[m, l, :] = x * np.sqrt((2*l - 1) / (l - m) * (2*l + 1) / (l + m)) * vdm[m, l-1, :] \
                            - np.sqrt((l + m - 1) / (l - m) * (2*l + 1) / (2*l - 3) * (l - m - 1) / (l + m)) * vdm[m, l-2, :]

    if norm == "schmidt":
        for l in range(0, nmax):
            if inverse:
                vdm[:, l, : ] = vdm[:, l, : ] * np.sqrt(2*l + 1)
            else:
                vdm[:, l, : ] = vdm[:, l, : ] / np.sqrt(2*l + 1)

    vdm = vdm[:mmax, :lmax]

    if csphase:
        for m in range(1, mmax, 2):
            vdm[m] *= -1

    return vdm

def _precompute_legpoly(mmax, lmax, t, norm="ortho", inverse=False, csphase=True):
    r"""
    Computes the values of (-1)^m c^l_m P^l_m(\cos \theta) at the positions specified by t (theta).
    The resulting tensor has shape (mmax, lmax, len(x)). The Condon-Shortley Phase (-1)^m
    can be turned off optionally.

    method of computation follows
    [1] Schaeffer, N.; Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Rapp, R.H.; A Fortran Program for the Computation of Gravimetric Quantities from High Degree Spherical Harmonic Expansions, Ohio State University Columbus; report; 1982;
        https://apps.dtic.mil/sti/citations/ADA123406
    [3] Schrama, E.; Orbit integration based upon interpolated gravitational gradients
    """

    return legpoly(mmax, lmax, np.cos(t), norm=norm, inverse=inverse, csphase=csphase)

def _precompute_dlegpoly(mmax, lmax, t, norm="ortho", inverse=False, csphase=True):
    r"""
    Computes the values of the derivatives $\frac{d}{d \theta} P^m_l(\cos \theta)$
    at the positions specified by t (theta), as well as $\frac{1}{\sin \theta} P^m_l(\cos \theta)$,
    needed for the computation of the vector spherical harmonics. The resulting tensor has shape
    (2, mmax, lmax, len(t)).

    computation follows
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    pct = _precompute_legpoly(mmax+1, lmax+1, t, norm=norm, inverse=inverse, csphase=False)

    dpct = np.zeros((2, mmax, lmax, len(t)), dtype=np.float64)

    # fill the derivative terms wrt theta
    for l in range(0, lmax):

        # m = 0
        dpct[0, 0, l] = - np.sqrt(l*(l+1)) * pct[1, l]

        # 0 < m < l
        for m in range(1, min(l, mmax)):
            dpct[0, m, l] = 0.5 * ( np.sqrt((l+m)*(l-m+1)) * pct[m-1, l] - np.sqrt((l-m)*(l+m+1)) * pct[m+1, l] )

        # m == l
        if mmax > l:
            dpct[0, l, l] = np.sqrt(l/2) * pct[l-1, l]

        # fill the - 1j m P^m_l / sin(phi). as this component is purely imaginary,
        # we won't store it explicitly in a complex array
        for m in range(1, min(l+1, mmax)):
            # this component is implicitly complex
            # we do not divide by m here as this cancels with the derivative of the exponential
            dpct[1, m, l] = 0.5 * np.sqrt((2*l+1)/(2*l+3)) * \
                ( np.sqrt((l-m+1)*(l-m+2)) * pct[m-1, l+1] + np.sqrt((l+m+1)*(l+m+2)) * pct[m+1, l+1] )

    if csphase:
        for m in range(1, mmax, 2):
            dpct[:, m] *= -1

    return dpct


#quadrature
import numpy as np


def _precompute_grid(n, grid="equidistant", a=0.0, b=1.0, periodic=False):

    if (grid != "equidistant") and periodic:
        raise ValueError(f"Periodic grid is only supported on equidistant grids.")

    # compute coordinates
    if grid == "equidistant":
        xlg, wlg = trapezoidal_weights(n, a=a, b=b, periodic=periodic)
    elif grid == "legendre-gauss":
        xlg, wlg = legendre_gauss_weights(n, a=a, b=b)
    elif grid == "lobatto":
        xlg, wlg = lobatto_weights(n, a=a, b=b)
    elif grid == "equiangular":
        xlg, wlg = clenshaw_curtiss_weights(n, a=a, b=b)
    else:
        raise ValueError(f"Unknown grid type {grid}")

    return xlg, wlg


def _precompute_latitudes(nlat, grid="equiangular"):
    r"""
    Convenience routine to precompute latitudes
    """

    # compute coordinates in the cosine theta domain
    xlg, wlg = _precompute_grid(nlat, grid=grid, a=-1.0, b=1.0, periodic=False)

    # to perform the quadrature and account for the jacobian of the sphere, the quadrature rule
    # is formulated in the cosine theta domain, which is designed to integrate functions of cos theta
    lats = np.flip(np.arccos(xlg)).copy()
    wlg = np.flip(wlg).copy()

    return lats, wlg


def trapezoidal_weights(n, a=-1.0, b=1.0, periodic=False):
    r"""
    Helper routine which returns equidistant nodes with trapezoidal weights
    on the interval [a, b]
    """

    xlg = np.linspace(a, b, n, endpoint=periodic)
    wlg = (b - a) / (n - periodic * 1) * np.ones(n)

    if not periodic:
        wlg[0] *= 0.5
        wlg[-1] *= 0.5

    return xlg, wlg


def legendre_gauss_weights(n, a=-1.0, b=1.0):
    r"""
    Helper routine which returns the Legendre-Gauss nodes and weights
    on the interval [a, b]
    """

    xlg, wlg = np.polynomial.legendre.leggauss(n)
    xlg = (b - a) * 0.5 * xlg + (b + a) * 0.5
    wlg = wlg * (b - a) * 0.5

    return xlg, wlg


def lobatto_weights(n, a=-1.0, b=1.0, tol=1e-16, maxiter=100):
    r"""
    Helper routine which returns the Legendre-Gauss-Lobatto nodes and weights
    on the interval [a, b]
    """

    wlg = np.zeros((n,))
    tlg = np.zeros((n,))
    tmp = np.zeros((n,))

    # Vandermonde Matrix
    vdm = np.zeros((n, n))

    # initialize Chebyshev nodes as first guess
    for i in range(n):
        tlg[i] = -np.cos(np.pi * i / (n - 1))

    tmp = 2.0

    for i in range(maxiter):
        tmp = tlg

        vdm[:, 0] = 1.0
        vdm[:, 1] = tlg

        for k in range(2, n):
            vdm[:, k] = ((2 * k - 1) * tlg * vdm[:, k - 1] - (k - 1) * vdm[:, k - 2]) / k

        tlg = tmp - (tlg * vdm[:, n - 1] - vdm[:, n - 2]) / (n * vdm[:, n - 1])

        if max(abs(tlg - tmp).flatten()) < tol:
            break

    wlg = 2.0 / ((n * (n - 1)) * (vdm[:, n - 1] ** 2))

    # rescale
    tlg = (b - a) * 0.5 * tlg + (b + a) * 0.5
    wlg = wlg * (b - a) * 0.5

    return tlg, wlg


def clenshaw_curtiss_weights(n, a=-1.0, b=1.0):
    r"""
    Computation of the Clenshaw-Curtis quadrature nodes and weights.
    This implementation follows

    [1] Joerg Waldvogel, Fast Construction of the Fejer and Clenshaw-Curtis Quadrature Rules; BIT Numerical Mathematics, Vol. 43, No. 1, pp. 001–018.
    """

    assert n > 1

    tcc = np.cos(np.linspace(np.pi, 0, n))

    if n == 2:
        wcc = np.array([1.0, 1.0])
    else:

        n1 = n - 1
        N = np.arange(1, n1, 2)
        l = len(N)
        m = n1 - l

        v = np.concatenate([2 / N / (N - 2), 1 / N[-1:], np.zeros(m)])
        v = 0 - v[:-1] - v[-1:0:-1]

        g0 = -np.ones(n1)
        g0[l] = g0[l] + n1
        g0[m] = g0[m] + n1
        g = g0 / (n1**2 - 1 + (n1 % 2))
        wcc = np.fft.ifft(v + g).real
        wcc = np.concatenate((wcc, wcc[:1]))

    # rescale
    tcc = (b - a) * 0.5 * tcc + (b + a) * 0.5
    wcc = wcc * (b - a) * 0.5

    return tcc, wcc


def fejer2_weights(n, a=-1.0, b=1.0):
    r"""
    Computation of the Fejer quadrature nodes and weights.
    This implementation follows

    [1] Joerg Waldvogel, Fast Construction of the Fejer and Clenshaw-Curtis Quadrature Rules; BIT Numerical Mathematics, Vol. 43, No. 1, pp. 001–018.
    """

    assert n > 2

    tcc = np.cos(np.linspace(np.pi, 0, n))

    n1 = n - 1
    N = np.arange(1, n1, 2)
    l = len(N)
    m = n1 - l

    v = np.concatenate([2 / N / (N - 2), 1 / N[-1:], np.zeros(m)])
    v = 0 - v[:-1] - v[-1:0:-1]

    wcc = np.fft.ifft(v).real
    wcc = np.concatenate((wcc, wcc[:1]))

    # rescale
    tcc = (b - a) * 0.5 * tcc + (b + a) * 0.5
    wcc = wcc * (b - a) * 0.5

    return tcc, wcc


#sht
class RealSHT(nn.Module):
    r"""
    Defines a module for computing the forward (real-valued) SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    The SHT is applied to the last two dimensions of the input

    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="equiangular", norm="ortho", csphase=True):
        r"""
        Initializes the SHT Layer, precomputing the necessary quadrature weights

        Parameters:
        nlat: input grid resolution in the latitudinal direction
        nlon: input grid resolution in the longitudinal direction
        grid: grid in the latitude direction (for now only tensor product grids are supported)
        """

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # TODO: include assertions regarding the dimensions

        # compute quadrature points and lmax based on the exactness of the quadrature
        if self.grid == "legendre-gauss":
            cost, w = legendre_gauss_weights(nlat, -1, 1)
            # maximum polynomial degree for Gauss Legendre is 2 * nlat - 1 >= 2 * lmax
            # and therefore lmax = nlat - 1 (inclusive)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, w = lobatto_weights(nlat, -1, 1)
            # maximum polynomial degree for Gauss Legendre is 2 * nlat - 3 >= 2 * lmax
            # and therefore lmax = nlat - 2 (inclusive)
            self.lmax = lmax or self.nlat - 1
        elif self.grid == "equiangular":
            cost, w = clenshaw_curtiss_weights(nlat, -1, 1)
            # in principle, Clenshaw-Curtiss quadrature is only exact up to polynomial degrees of nlat
            # however, we observe that the quadrature is remarkably accurate for higher degress. This is why we do not
            # choose a lower lmax for now.
            self.lmax = lmax or self.nlat
        else:
            raise (ValueError("Unknown quadrature mode"))

        # apply cosine transform and flip them
        tq = np.flip(np.arccos(cost))

        # determine the dimensions
        self.mmax = mmax or self.nlon // 2 + 1

        # combine quadrature weights with the legendre weights
        weights = torch.from_numpy(w)
        pct = _precompute_legpoly(self.mmax, self.lmax, tq, norm=self.norm, csphase=self.csphase)
        pct = torch.from_numpy(pct)
        weights = torch.einsum("mlk,k->mlk", pct, weights)

        # remember quadrature weights
        self.register_buffer("weights", weights, persistent=False)

    def extra_repr(self):
        r"""
        Pretty print module
        """
        return f"nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}"

    def forward(self, x: torch.Tensor):

        if x.dim() < 2:
            raise ValueError(f"Expected tensor with at least 2 dimensions but got {x.dim()} instead")

        assert x.shape[-2] == self.nlat
        assert x.shape[-1] == self.nlon

        # apply real fft in the longitudinal direction
        x = 2.0 * torch.pi * torch.fft.rfft(x, dim=-1, norm="forward")

        # do the Legendre-Gauss quadrature
        x = torch.view_as_real(x)

        # distributed contraction: fork
        out_shape = list(x.size())
        out_shape[-3] = self.lmax
        out_shape[-2] = self.mmax
        xout = torch.zeros(out_shape, dtype=x.dtype, device=x.device)

        # contraction
        xout[..., 0] = torch.einsum("...km,mlk->...lm", x[..., : self.mmax, 0], self.weights.to(x.dtype))
        xout[..., 1] = torch.einsum("...km,mlk->...lm", x[..., : self.mmax, 1], self.weights.to(x.dtype))
        x = torch.view_as_complex(xout)

        return x


class InverseRealSHT(nn.Module):
    r"""
    Defines a module for computing the inverse (real-valued) SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    nlat, nlon: Output dimensions
    lmax, mmax: Input dimensions (spherical coefficients). For convenience, these are inferred from the output dimensions

    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="equiangular", norm="ortho", csphase=True):

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, _ = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, _ = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat - 1
        elif self.grid == "equiangular":
            cost, _ = clenshaw_curtiss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        else:
            raise (ValueError("Unknown quadrature mode"))

        # apply cosine transform and flip them
        t = np.flip(np.arccos(cost))

        # determine the dimensions
        self.mmax = mmax or self.nlon // 2 + 1

        pct = _precompute_legpoly(self.mmax, self.lmax, t, norm=self.norm, inverse=True, csphase=self.csphase)
        pct = torch.from_numpy(pct)

        # register buffer
        self.register_buffer("pct", pct, persistent=False)

    def extra_repr(self):
        r"""
        Pretty print module
        """
        return f"nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}"

    def forward(self, x: torch.Tensor):

        if len(x.shape) < 2:
            raise ValueError(f"Expected tensor with at least 2 dimensions but got {len(x.shape)} instead")

        assert x.shape[-2] == self.lmax
        assert x.shape[-1] == self.mmax

        # Evaluate associated Legendre functions on the output nodes
        x = torch.view_as_real(x)
        
        # print("pct.requires_grad =", self.pct.requires_grad)
        # print("pct.shape =", self.pct.shape, "version =", self.pct._version)

        # print(self.pct._version,'before real')
        rl = torch.einsum("...lm, mlk->...km", x[..., 0], self.pct)
        # print(self.pct._version,'after real')
        im = torch.einsum("...lm, mlk->...km", x[..., 1],self.pct)
        # print(self.pct._version,'after im')
        # print(x[...,1].shape , x[...,1]._version, self.pct.shape, self.pct._version)
        xs = torch.stack((rl, im), -1)

    
        # print(self.pct._version,'before real')
        # xs = torch.einsum("...lmr, mlk->...kmr", x, self.pct.detach()).contiguous()
        # print(self.pct._version,'after real')
        # apply the inverse (real) FFT
        x = torch.view_as_complex(xs)
        # x[..., 0].imag = 0.0
        # if (self.nlon % 2 == 0) and (self.nlon // 2 < self.mmax):
        #     x[..., self.nlon // 2].imag = 0.0
        x = torch.fft.irfft(x, n=self.nlon, dim=-1, norm="forward")
        # print(x.shape, x._version)
        return x


#layers
import torch
import torch.nn as nn
import torch.fft
from torch.utils.checkpoint import checkpoint
import math

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
    tensor: an n-dimensional `torch.Tensor`
    mean: the mean of the normal distribution
    std: the standard deviation of the normal distribution
    a: the minimum cutoff value
    b: the maximum cutoff value
    Examples:
    >>> w = torch.empty(3, 5)
    >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


@torch.jit.script
def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1. - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2d ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features = None,
                 out_features = None,
                 act_layer = nn.ReLU,
                 output_bias = False,
                 drop_rate = 0.,
                 checkpointing = False,
                 gain = 1.0):
        super(MLP, self).__init__()
        self.checkpointing = checkpointing
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Fist dense layer
        fc1 = nn.Conv2d(in_features, hidden_features, 1, bias=True)
        # initialize the weights correctly
        scale = math.sqrt(2.0 / in_features)
        nn.init.normal_(fc1.weight, mean=0., std=scale)
        if fc1.bias is not None:
            nn.init.constant_(fc1.bias, 0.0)

        # activation
        act = act_layer()

        # output layer
        fc2 = nn.Conv2d(hidden_features, out_features, 1, bias=output_bias)
        # gain factor for the output determines the scaling of the output init
        scale = math.sqrt(gain / hidden_features)
        nn.init.normal_(fc2.weight, mean=0., std=scale)
        if fc2.bias is not None:
            nn.init.constant_(fc2.bias, 0.0)

        if drop_rate > 0.:
            drop = nn.Dropout2d(drop_rate)
            self.fwd = nn.Sequential(fc1, act, drop, fc2, drop)
        else:
            self.fwd = nn.Sequential(fc1, act, fc2)

    @torch.jit.ignore
    def checkpoint_forward(self, x):
        return checkpoint(self.fwd, x)

    def forward(self, x):
        if self.checkpointing:
            return self.checkpoint_forward(x)
        else:
            return self.fwd(x)

class RealFFT2(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """
    def __init__(self,
                 nlat,
                 nlon,
                 lmax = None,
                 mmax = None):
        super(RealFFT2, self).__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.lmax = lmax or self.nlat
        self.mmax = mmax or self.nlon // 2 + 1

    def forward(self, x):
        y = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        y = torch.cat((y[..., :math.ceil(self.lmax/2), :self.mmax], y[..., -math.floor(self.lmax/2):, :self.mmax]), dim=-2)
        return y

class InverseRealFFT2(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """
    def __init__(self,
                 nlat,
                 nlon,
                 lmax = None,
                 mmax = None):
        super(InverseRealFFT2, self).__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.lmax = lmax or self.nlat
        self.mmax = mmax or self.nlon // 2 + 1

    def forward(self, x):
        return torch.fft.irfft2(x, dim=(-2, -1), s=(self.nlat, self.nlon), norm="ortho")

class SpectralConvS2(nn.Module):
    """
    Spectral Convolution according to Driscoll & Healy. Designed for convolutions on the two-sphere S2
    using the Spherical Harmonic Transforms in torch-harmonics, but supports convolutions on the periodic
    domain via the RealFFT2 and InverseRealFFT2 wrappers.
    """

    def __init__(self,
                 forward_transform,
                 inverse_transform,
                 in_channels,
                 out_channels,
                 gain = 2.,
                 operator_type = "driscoll-healy",
                 lr_scale_exponent = 0,
                 bias = False):
        super().__init__()

        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform

        self.modes_lat = self.inverse_transform.lmax
        self.modes_lon = self.inverse_transform.mmax

        self.scale_residual = (self.forward_transform.nlat != self.inverse_transform.nlat) \
                        or (self.forward_transform.nlon != self.inverse_transform.nlon)

        # remember factorization details
        self.operator_type = operator_type

        assert self.inverse_transform.lmax == self.modes_lat
        assert self.inverse_transform.mmax == self.modes_lon

        weight_shape = [out_channels, in_channels]

        if self.operator_type == "diagonal":
            weight_shape += [self.modes_lat, self.modes_lon]
            self.contract_func = "...ilm,oilm->...olm"
        elif self.operator_type == "block-diagonal":
            weight_shape += [self.modes_lat, self.modes_lon, self.modes_lon]
            self.contract_func = "...ilm,oilnm->...oln"
        elif self.operator_type == "driscoll-healy":
            weight_shape += [self.modes_lat]
            self.contract_func = "...ilm,oil->...olm"
        else:
            raise NotImplementedError(f"Unkonw operator type f{self.operator_type}")

        # form weight tensors
        scale = math.sqrt(gain / in_channels)
        self.weight = nn.Parameter(scale * torch.randn(*weight_shape, dtype=torch.complex64))
        if bias:
            self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))


    def forward(self, x):

        dtype = x.dtype
        x = x.float()
        residual = x

        with torch.autocast(device_type="cuda", enabled=False):
            x = self.forward_transform(x)
            if self.scale_residual:
                residual = self.inverse_transform(x)

        x = torch.einsum(self.contract_func, x, self.weight)

        with torch.autocast(device_type="cuda", enabled=False):
            x = self.inverse_transform(x)

        if hasattr(self, "bias"):
            x = x + self.bias
        x = x.type(dtype)

        return x, residual


#sfno class
import torch
import torch.nn as nn
from functools import partial
class SphericalFourierNeuralOperatorBlock(nn.Module):
    """
    Helper module for a single SFNO/FNO block. Can use both FFTs and SHTs to represent either FNO or SFNO blocks.
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        input_dim,
        output_dim,
        operator_type="driscoll-healy",
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path=0.0,
        act_layer=nn.ReLU,
        norm_layer=nn.Identity,
        factorization=None,
        separable=False,
        rank=128,
        inner_skip="linear",
        outer_skip=None,
        use_mlp=True,
    ):
        super().__init__()

        if act_layer == nn.Identity:
            gain_factor = 1.0
        else:
            gain_factor = 2.0

        if inner_skip == "linear" or inner_skip == "identity":
            gain_factor /= 2.0

        self.global_conv = SpectralConvS2(forward_transform, inverse_transform, input_dim, output_dim, gain=gain_factor, operator_type=operator_type, bias=False)

        if inner_skip == "linear":
            self.inner_skip = nn.Conv2d(input_dim, output_dim, 1, 1)
            nn.init.normal_(self.inner_skip.weight, std=math.sqrt(gain_factor / input_dim))
        elif inner_skip == "identity":
            assert input_dim == output_dim
            self.inner_skip = nn.Identity()
        elif inner_skip == "none":
            pass
        else:
            raise ValueError(f"Unknown skip connection type {inner_skip}")

        # first normalisation layer
        self.norm0 = norm_layer()

        # dropout
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        gain_factor = 1.0
        if outer_skip == "linear" or inner_skip == "identity":
            gain_factor /= 2.0

        if use_mlp == True:
            mlp_hidden_dim = int(output_dim * mlp_ratio)
            self.mlp = MLP(
                in_features=output_dim, out_features=input_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop_rate=drop_rate, checkpointing=False, gain=gain_factor
            )

        if outer_skip == "linear":
            self.outer_skip = nn.Conv2d(input_dim, input_dim, 1, 1)
            torch.nn.init.normal_(self.outer_skip.weight, std=math.sqrt(gain_factor / input_dim))
        elif outer_skip == "identity":
            assert input_dim == output_dim
            self.outer_skip = nn.Identity()
        elif outer_skip == "none":
            pass
        else:
            raise ValueError(f"Unknown skip connection type {outer_skip}")

        # second normalisation layer
        self.norm1 = norm_layer()

    def forward(self, x):

        x, residual = self.global_conv(x)

        x = self.norm0(x)

        if hasattr(self, "inner_skip"):
            x = x + self.inner_skip(residual)

        if hasattr(self, "mlp"):
            x = self.mlp(x)

        x = self.norm1(x)

        x = self.drop_path(x)

        if hasattr(self, "outer_skip"):
            x = x + self.outer_skip(residual)

        return x


class SphericalFourierNeuralOperatorNet(nn.Module):
    """
    SphericalFourierNeuralOperator module. Implements the 'linear' variant of the Spherical Fourier Neural Operator
    as presented in [1]. Spherical convolutions are applied via spectral transforms to apply a geometrically consistent
    and approximately equivariant architecture.

    Parameters
    ----------
    img_shape : tuple, optional
        Shape of the input channels, by default (128, 256)
    operator_type : str, optional
        Type of operator to use ('driscoll-healy', 'diagonal'), by default "driscoll-healy"
    scale_factor : int, optional
        Scale factor to use, by default 3
    in_chans : int, optional
        Number of input channels, by default 3
    out_chans : int, optional
        Number of output channels, by default 3
    embed_dim : int, optional
        Dimension of the embeddings, by default 256
    num_layers : int, optional
        Number of layers in the network, by default 4
    activation_function : str, optional
        Activation function to use, by default "gelu"
    encoder_layers : int, optional
        Number of layers in the encoder, by default 1
    use_mlp : int, optional
        Whether to use MLPs in the SFNO blocks, by default True
    mlp_ratio : int, optional
        Ratio of MLP to use, by default 2.0
    drop_rate : float, optional
        Dropout rate, by default 0.0
    drop_path_rate : float, optional
        Dropout path rate, by default 0.0
    normalization_layer : str, optional
        Type of normalization layer to use ("layer_norm", "instance_norm", "none"), by default "instance_norm"
    hard_thresholding_fraction : float, optional
        Fraction of hard thresholding (frequency cutoff) to apply, by default 1.0
    big_skip : bool, optional
        Whether to add a single large skip connection, by default True
    pos_embed : bool, optional
        Whether to use positional embedding, by default True

    Example:
    --------
    >>> model = SphericalFourierNeuralOperatorNet(
    ...         img_shape=(128, 256),
    ...         scale_factor=4,
    ...         in_chans=2,
    ...         out_chans=2,
    ...         embed_dim=16,
    ...         num_layers=4,
    ...         use_mlp=True,)
    >>> model(torch.randn(1, 2, 128, 256)).shape
    torch.Size([1, 2, 128, 256])

    References
    -----------
    .. [1] Bonev B., Kurth T., Hundt C., Pathak, J., Baust M., Kashinath K., Anandkumar A.;
        "Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere" (2023).
        ICML 2023, https://arxiv.org/abs/2306.03838.
    """

    def __init__(
        self,
        img_size=(128, 256),
        operator_type="driscoll-healy",
        grid="equiangular",
        grid_internal="legendre-gauss",
        scale_factor=3,
        in_chans=3,
        out_chans=3,
        embed_dim=256,
        num_layers=4,
        activation_function="relu",
        encoder_layers=1,
        use_mlp=True,
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        normalization_layer="none",
        hard_thresholding_fraction=1.0,
        use_complex_kernels=True,
        big_skip=False,
        pos_embed=False,
    ):

        super().__init__()

        self.operator_type = operator_type
        self.img_size = img_size
        self.grid = grid
        self.grid_internal = grid_internal
        self.scale_factor = scale_factor
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.normalization_layer = normalization_layer
        self.use_mlp = use_mlp
        self.encoder_layers = encoder_layers
        self.big_skip = big_skip

        # activation function
        if activation_function == "relu":
            self.activation_function = nn.ReLU
        elif activation_function == "gelu":
            self.activation_function = nn.GELU
        # for debugging purposes
        elif activation_function == "identity":
            self.activation_function = nn.Identity
        else:
            raise ValueError(f"Unknown activation function {activation_function}")

        # compute downsampled image size. We assume that the latitude-grid includes both poles
        self.h = (self.img_size[0] - 1) // scale_factor + 1
        self.w = self.img_size[1] // scale_factor

        # dropout
        self.pos_drop = nn.Dropout(p=drop_rate) if drop_rate > 0.0 else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]

        # pick norm layer
        if self.normalization_layer == "layer_norm":
            norm_layer0 = partial(nn.LayerNorm, normalized_shape=(self.img_size[0], self.img_size[1]), eps=1e-6)
            norm_layer1 = partial(nn.LayerNorm, normalized_shape=(self.h, self.w), eps=1e-6)
        elif self.normalization_layer == "instance_norm":
            norm_layer0 = partial(nn.InstanceNorm2d, num_features=self.embed_dim, eps=1e-6, affine=True, track_running_stats=False)
            norm_layer1 = partial(nn.InstanceNorm2d, num_features=self.embed_dim, eps=1e-6, affine=True, track_running_stats=False)
        elif self.normalization_layer == "none":
            norm_layer0 = nn.Identity
            norm_layer1 = norm_layer0
        else:
            raise NotImplementedError(f"Error, normalization {self.normalization_layer} not implemented.")

        if pos_embed == "latlon" or pos_embed == True:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, self.img_size[0], self.img_size[1]))
            nn.init.constant_(self.pos_embed, 0.0)
        elif pos_embed == "lat":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, self.img_size[0], 1))
            nn.init.constant_(self.pos_embed, 0.0)
        elif pos_embed == "const":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, 1, 1))
            nn.init.constant_(self.pos_embed, 0.0)
        else:
            self.pos_embed = None

        # construct an encoder with num_encoder_layers
        num_encoder_layers = 1
        encoder_hidden_dim = int(self.embed_dim * mlp_ratio)
        current_dim = self.in_chans
        encoder_layers = []
        for l in range(num_encoder_layers - 1):
            fc = nn.Conv2d(current_dim, encoder_hidden_dim, 1, bias=True)
            # initialize the weights correctly
            scale = math.sqrt(2.0 / current_dim)
            nn.init.normal_(fc.weight, mean=0.0, std=scale)
            if fc.bias is not None:
                nn.init.constant_(fc.bias, 0.0)
            encoder_layers.append(fc)
            encoder_layers.append(self.activation_function())
            current_dim = encoder_hidden_dim
        fc = nn.Conv2d(current_dim, self.embed_dim, 1, bias=False)
        scale = math.sqrt(1.0 / current_dim)
        nn.init.normal_(fc.weight, mean=0.0, std=scale)
        if fc.bias is not None:
            nn.init.constant_(fc.bias, 0.0)
        encoder_layers.append(fc)
        self.encoder = nn.Sequential(*encoder_layers)

        # compute the modes for the sht
        modes_lat = self.h
        # due to some spectral artifacts with cufft, we substract one mode here
        modes_lon = (self.w // 2 + 1) -1

        modes_lat = modes_lon = int(min(modes_lat, modes_lon) * self.hard_thresholding_fraction)

        self.trans_down = RealSHT(*self.img_size, lmax=modes_lat, mmax=modes_lon, grid=self.grid).float()
        self.itrans_up = InverseRealSHT(*self.img_size, lmax=modes_lat, mmax=modes_lon, grid=self.grid).float()
        self.trans = RealSHT(self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid=grid_internal).float()
        self.itrans = InverseRealSHT(self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid=grid_internal).float()

        self.blocks = nn.ModuleList([])
        for i in range(self.num_layers):

            first_layer = i == 0
            last_layer = i == self.num_layers - 1

            forward_transform = self.trans_down if first_layer else self.trans
            inverse_transform = self.itrans_up if last_layer else self.itrans

            inner_skip = "none"
            outer_skip = "identity"

            if first_layer:
                norm_layer = norm_layer1
            elif last_layer:
                norm_layer = norm_layer0
            else:
                norm_layer = norm_layer1

            block = SphericalFourierNeuralOperatorBlock(
                forward_transform,
                inverse_transform,
                self.embed_dim,
                self.embed_dim,
                operator_type=self.operator_type,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                drop_path=dpr[i],
                act_layer=self.activation_function,
                norm_layer=norm_layer,
                inner_skip=inner_skip,
                outer_skip=outer_skip,
                use_mlp=use_mlp,
            )

            self.blocks.append(block)

        # construct an decoder with num_decoder_layers
        num_decoder_layers = 1
        decoder_hidden_dim = int(self.embed_dim * mlp_ratio)
        current_dim = self.embed_dim + self.big_skip * self.in_chans
        decoder_layers = []
        for l in range(num_decoder_layers - 1):
            fc = nn.Conv2d(current_dim, decoder_hidden_dim, 1, bias=True)
            # initialize the weights correctly
            scale = math.sqrt(2.0 / current_dim)
            nn.init.normal_(fc.weight, mean=0.0, std=scale)
            if fc.bias is not None:
                nn.init.constant_(fc.bias, 0.0)
            decoder_layers.append(fc)
            decoder_layers.append(self.activation_function())
            current_dim = decoder_hidden_dim
        fc = nn.Conv2d(current_dim, self.out_chans, 1, bias=False)
        scale = math.sqrt(1.0 / current_dim)
        nn.init.normal_(fc.weight, mean=0.0, std=scale)
        if fc.bias is not None:
            nn.init.constant_(fc.bias, 0.0)
        decoder_layers.append(fc)
        self.decoder = nn.Sequential(*decoder_layers)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward_features(self, x):

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = checkpoint(blk, x, use_reentrant = False)

        return x

    def forward(self, x):

        if self.big_skip:
            residual = x

        x = self.encoder(x)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        x = self.forward_features(x)

        if self.big_skip:
            x = torch.cat((x, residual), dim=1)

        x = self.decoder(x)

        return x














































