"""
======================
Cosmology applications
======================
"""

from .mcfit import mcfit
from .kernels import *
from numpy import pi, real_if_close


class P2xi(mcfit):
    """
    Power spectrum to correlation function
    """
    def __init__(self, k, l=0, q=1.5, **kwargs):
        self.l = l
        UK = Mellin_SphericalBesselJ(l)
        prefac = real_if_close(1j**l) * x**3 / (2*pi)**1.5
        postfac = 1
        mcfit.__init__(self, k, UK, q, prefac=prefac, postfac=postfac, **kwargs)


class xi2P(mcfit):
    """
    Correlation function to power spectrum
    Also radial profile to its Fourier transform
    """
    def __init__(self, r, l=0, q=1.5, **kwargs):
        self.l = l
        UK = Mellin_SphericalBesselJ(l)
        prefac = x**3
        postfac = (2*pi)**1.5 / real_if_close(1j**l)
        mcfit.__init__(self, r, UK, q, prefac=prefac, postfac=postfac, **kwargs)


class TophatVar(mcfit):
    """
    Variance in a top-hat window
    """
    def __init__(self, k, q=1.5, **kwargs):
        UK = Mellin_TophatSq(3)
        prefac = k**3 / (2 * pi**2)
        postfac = 1
        mcfit.__init__(self, k, UK, q, prefac=prefac, postfac=postfac, **kwargs)


class GaussVar(mcfit):
    """
    Variance in a Gaussian window
    """
    def __init__(self, k, q=1.5, **kwargs):
        UK = Mellin_GaussSq()
        prefac = k**3 / (2 * pi**2)
        postfac = 1
        mcfit.__init__(self, k, UK, q, prefac=prefac, postfac=postfac, **kwargs)


class ExcursionSet(mcfit):
    """
    Excursion set trajectory
    BCEK 91 model
    """
    pass
