"""
===========================================
Common Integral transforms and applications
===========================================
"""

from .fit import fit
from .kernels import *
from numpy import pi
from scipy.special import gamma


class Hankel(fit):
    """
    Hankel transform pair
    """
    def __init__(self, x, nu=0, q=1, **kwargs):
        self.nu = nu
        UK = Mellin_BesselJ(nu)
        prefac = x**2
        fit.__init__(self, x, UK, q, prefac=prefac, **kwargs)


class SphericalBessel(fit):
    """
    Spherical Bessel transform pair
    """
    def __init__(self, x, nu=0, q=1.5, **kwargs):
        self.nu = nu
        UK = Mellin_SphericalBesselJ(nu)
        prefac = x**3
        fit.__init__(self, x, UK, q, prefac=prefac, **kwargs)


class FourierSine(fit):
    """
    Fourier sine transform pair
    """
    def __init__(self, x, q=0.5, **kwargs):
        UK = Mellin_FourierSine()
        prefac = x
        fit.__init__(self, x, UK, q, prefac=prefac, **kwargs)


class FourierCosine(fit):
    """
    Fourier cosine transform pair
    """
    def __init__(self, x, q=0.5, **kwargs):
        UK = Mellin_FourierCosine()
        prefac = x
        fit.__init__(self, x, UK, q, prefac=prefac, **kwargs)


class TophatSmooth(fit):
    """
    Top-hat smoothing of a radial function
    """
    def __init__(self, x, d=3, q=0, **kwargs):
        self.d = d
        UK = Mellin_Tophat(d)
        prefac = x**d / (2**(d-1) * pi**(d/2) * gamma(d/2))
        fit.__init__(self, x, UK, q, prefac=prefac, **kwargs)


class GaussSmooth(fit):
    """
    Gaussian smoothing of a radial function
    """
    def __init__(self, x, d=3, q=0, **kwargs):
        self.d = d
        UK = Mellin_Gauss()
        prefac = x**d / (2**(d-1) * pi**(d/2) * gamma(d/2))
        fit.__init__(self, x, UK, q, prefac=prefac, **kwargs)
