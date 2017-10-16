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


__all__ = ['Hankel', 'SphericalBessel', 'FourierSine', 'FourierCosine',
           'TophatSmooth', 'GaussSmooth']


class Hankel(mcfit):
    """Hankel transform pair
    """
    def __init__(self, x, nu=0, q=1, N=None, lowring=True):
        self.nu = nu
        UK = kernels.Mellin_BesselJ(nu)
        mcfit.__init__(self, x, UK, q, N=N, lowring=lowring)
        self.prefac *= self.x**2


class SphericalBessel(mcfit):
    """Spherical Bessel transform pair
    """
    def __init__(self, x, nu=0, q=1.5, N=None, lowring=True):
        self.nu = nu
        UK = kernels.Mellin_SphericalBesselJ(nu)
        mcfit.__init__(self, x, UK, q, N=N, lowring=lowring)
        self.prefac *= self.x**3


class FourierSine(mcfit):
    """Fourier sine transform pair
    """
    def __init__(self, x, q=0.5, N=None, lowring=True):
        UK = kernels.Mellin_FourierSine()
        mcfit.__init__(self, x, UK, q, N=N, lowring=lowring)
        self.prefac *= self.x


class FourierCosine(mcfit):
    """Fourier cosine transform pair
    """
    def __init__(self, x, q=0.5, N=None, lowring=True):
        UK = kernels.Mellin_FourierCosine()
        mcfit.__init__(self, x, UK, q, N=N, lowring=lowring)
        self.prefac *= self.x


class TophatSmooth(mcfit):
    """Top-hat smoothing of a radial function
    """
    def __init__(self, x, d=3, q=0, N=None, lowring=True):
        self.d = d
        UK = kernels.Mellin_Tophat(d)
        mcfit.__init__(self, x, UK, q, N=N, lowring=lowring)
        self.prefac *= self.x**d / (2**(d-1) * numpy.pi**(d/2) * gamma(d/2))


class GaussSmooth(mcfit):
    """Gaussian smoothing of a radial function
    """
    def __init__(self, x, d=3, q=0, N=None, lowring=True):
        self.d = d
        UK = kernels.Mellin_Gauss()
        mcfit.__init__(self, x, UK, q, N=N, lowring=lowring)
        self.prefac *= self.x**d / (2**(d-1) * numpy.pi**(d/2) * gamma(d/2))
