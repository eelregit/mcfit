"""
===========================================
Common Integral transforms and applications
===========================================
"""

from .mcfit import mcfit
from . import kernels
from numpy import pi
from scipy.special import gamma


__all__ = ['Hankel', 'SphericalBessel', 'FourierSine', 'FourierCosine',
           'TophatSmooth', 'GaussSmooth']


class Hankel(mcfit):
    """
    Hankel transform pair
    """
    def __init__(self, x, nu=0, q=1, N=None, lowring=True):
        self.nu = nu
        UK = kernels.Mellin_BesselJ(nu)
        prefac = x**2
        postfac = 1
        mcfit.__init__(self, x, UK, q, N=N, lowring=lowring, prefac=prefac, postfac=postfac)


class SphericalBessel(mcfit):
    """
    Spherical Bessel transform pair
    """
    def __init__(self, x, nu=0, q=1.5, N=None, lowring=True):
        self.nu = nu
        UK = kernels.Mellin_SphericalBesselJ(nu)
        prefac = x**3
        postfac = 1
        mcfit.__init__(self, x, UK, q, N=N, lowring=lowring, prefac=prefac, postfac=postfac)


class FourierSine(mcfit):
    """
    Fourier sine transform pair
    """
    def __init__(self, x, q=0.5, N=None, lowring=True):
        UK = kernels.Mellin_FourierSine()
        prefac = x
        postfac = 1
        mcfit.__init__(self, x, UK, q, N=N, lowring=lowring, prefac=prefac, postfac=postfac)


class FourierCosine(mcfit):
    """
    Fourier cosine transform pair
    """
    def __init__(self, x, q=0.5, N=None, lowring=True):
        UK = kernels.Mellin_FourierCosine()
        prefac = x
        postfac = 1
        mcfit.__init__(self, x, UK, q, N=N, lowring=lowring, prefac=prefac, postfac=postfac)


class TophatSmooth(mcfit):
    """
    Top-hat smoothing of a radial function
    """
    def __init__(self, x, d=3, q=0, N=None, lowring=True):
        self.d = d
        UK = kernels.Mellin_Tophat(d)
        prefac = x**d / (2**(d-1) * pi**(d/2) * gamma(d/2))
        postfac = 1
        mcfit.__init__(self, x, UK, q, N=N, lowring=lowring, prefac=prefac, postfac=postfac)


class GaussSmooth(mcfit):
    """
    Gaussian smoothing of a radial function
    """
    def __init__(self, x, d=3, q=0, N=None, lowring=True):
        self.d = d
        UK = kernels.Mellin_Gauss()
        prefac = x**d / (2**(d-1) * pi**(d/2) * gamma(d/2))
        postfac = 1
        mcfit.__init__(self, x, UK, q, N=N, lowring=lowring, prefac=prefac, postfac=postfac)
