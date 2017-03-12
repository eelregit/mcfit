"""
===========================================
Common Integral transforms and applications
===========================================
"""

from __future__ import division
from .mcfit import mcfit
from .kernels import *
from numpy import pi
from scipy.special import gamma


class Hankel(mcfit):
    """
    Hankel transform pair
    """
    def __init__(self, x, nu=0, q=1, **kwargs):
        self.nu = nu
        UK = Mellin_BesselJ(nu)
        prefac = x**2
        postfac = 1
        mcfit.__init__(self, x, UK, q, prefac=prefac, postfac=postfac, **kwargs)


class SphericalBessel(mcfit):
    """
    Spherical Bessel transform pair
    """
    def __init__(self, x, nu=0, q=1.5, **kwargs):
        self.nu = nu
        UK = Mellin_SphericalBesselJ(nu)
        prefac = x**3
        postfac = 1
        mcfit.__init__(self, x, UK, q, prefac=prefac, postfac=postfac, **kwargs)


class FourierSine(mcfit):
    """
    Fourier sine transform pair
    """
    def __init__(self, x, q=0.5, **kwargs):
        UK = Mellin_FourierSine()
        prefac = x
        postfac = 1
        mcfit.__init__(self, x, UK, q, prefac=prefac, postfac=postfac, **kwargs)


class FourierCosine(mcfit):
    """
    Fourier cosine transform pair
    """
    def __init__(self, x, q=0.5, **kwargs):
        UK = Mellin_FourierCosine()
        prefac = x
        postfac = 1
        mcfit.__init__(self, x, UK, q, prefac=prefac, postfac=postfac, **kwargs)


class TophatSmooth(mcfit):
    """
    Top-hat smoothing of a radial function
    """
    def __init__(self, x, d=3, q=0, **kwargs):
        self.d = d
        UK = Mellin_Tophat(d)
        prefac = x**d / (2**(d-1) * pi**(d/2) * gamma(d/2))
        postfac = 1
        mcfit.__init__(self, x, UK, q, prefac=prefac, postfac=postfac, **kwargs)


class GaussSmooth(mcfit):
    """
    Gaussian smoothing of a radial function
    """
    def __init__(self, x, d=3, q=0, **kwargs):
        self.d = d
        UK = Mellin_Gauss()
        prefac = x**d / (2**(d-1) * pi**(d/2) * gamma(d/2))
        postfac = 1
        mcfit.__init__(self, x, UK, q, prefac=prefac, postfac=postfac, **kwargs)
