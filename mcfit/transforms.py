"""
===========================================
Common Integral transforms and applications
===========================================
"""

from __future__ import division
from .mcfit import mcfit
from . import kernels
import numpy
from scipy.special import gamma


__all__ = ['Hankel', 'SphericalBessel', 'DoubleBessel', 'DoubleSphericalBessel',
            'FourierSine', 'FourierCosine', 'TophatSmooth', 'GaussSmooth']


class Hankel(mcfit):
    """Hankel transform pair.
    """
    def __init__(self, x, nu=0, q=1, **kwargs):
        self.nu = nu
        UK = kernels.Mellin_BesselJ(nu)
        mcfit.__init__(self, x, UK, q, **kwargs)
        self.prefac *= self.x**2


class SphericalBessel(mcfit):
    """Spherical Bessel transform pair.
    """
    def __init__(self, x, nu=0, q=1.5, **kwargs):
        self.nu = nu
        UK = kernels.Mellin_SphericalBesselJ(nu)
        mcfit.__init__(self, x, UK, q, **kwargs)
        self.prefac *= self.x**3


class FourierSine(mcfit):
    """Fourier sine transform pair.
    """
    def __init__(self, x, q=0.5, **kwargs):
        UK = kernels.Mellin_FourierSine()
        mcfit.__init__(self, x, UK, q, **kwargs)
        self.prefac *= self.x


class FourierCosine(mcfit):
    """Fourier cosine transform pair.
    """
    def __init__(self, x, q=0.5, **kwargs):
        UK = kernels.Mellin_FourierCosine()
        mcfit.__init__(self, x, UK, q, **kwargs)
        self.prefac *= self.x


class DoubleBessel(mcfit):
    r"""Compute integrals with two Bessel functions.

    .. math:: G(y_1; \alpha) \equiv G(y_1, y_2=\alpha y_1)
                = \int_0^\infty F(x) J_{\nu_1}(xy_1) J_{\nu_2}(xy_2) \,x\d x

    Parameters
    ----------
    alpha : float
        y2 / y1
    nu : float, optional
        default is 0
    nu1 : float, optional
        default is nu
    nu2 : float, optional
        default is nu
    """
    def __init__(self, x, alpha, nu=0, nu1=None, nu2=None, q=None, **kwargs):
        self.alpha = alpha
        if nu1 is None:
            nu1 = nu
        if nu2 is None:
            nu2 = nu
        self.nu1 = nu1
        self.nu2 = nu2
        UK = kernels.Mellin_DoubleBesselJ(alpha, nu1, nu2)
        if q is None:
            q = 1
            if alpha == 1:
                q = 0.5
        mcfit.__init__(self, x, UK, q, **kwargs)
        self.prefac *= self.x**2


class DoubleSphericalBessel(mcfit):
    r"""Compute integrals with two spherical Bessel functions.

    .. math:: G(y_1; \alpha) \equiv G(y_1, y_2=\alpha y_1)
                = \int_0^\infty F(x) j_{\nu_1}(xy_1) j_{\nu_2}(xy_2) \,x^2\d x

    Parameters
    ----------
    alpha : float
        y2 / y1
    nu : float, optional
        default is 0
    nu1 : float, optional
        default is nu
    nu2 : float, optional
        default is nu
    """
    def __init__(self, x, alpha, nu=0, nu1=None, nu2=None, q=None, **kwargs):
        self.alpha = alpha
        if nu1 is None:
            nu1 = nu
        if nu2 is None:
            nu2 = nu
        self.nu1 = nu1
        self.nu2 = nu2
        UK = kernels.Mellin_DoubleSphericalBesselJ(alpha, nu1, nu2)
        if q is None:
            q = 2
            if alpha == 1:
                q = 1.5
        mcfit.__init__(self, x, UK, q, **kwargs)
        self.prefac *= self.x**3


class TophatSmooth(mcfit):
    """Top-hat smoothing of a radial function.
    """
    def __init__(self, x, d=3, q=0, **kwargs):
        self.d = d
        UK = kernels.Mellin_Tophat(d)
        mcfit.__init__(self, x, UK, q, **kwargs)
        self.prefac *= self.x**d / (2**(d-1) * numpy.pi**(d/2) * gamma(d/2))


class GaussSmooth(mcfit):
    """Gaussian smoothing of a radial function.
    """
    def __init__(self, x, d=3, q=0, **kwargs):
        self.d = d
        UK = kernels.Mellin_Gauss()
        mcfit.__init__(self, x, UK, q, **kwargs)
        self.prefac *= self.x**d / (2**(d-1) * numpy.pi**(d/2) * gamma(d/2))
