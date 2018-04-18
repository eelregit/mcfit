"""
======================
Cosmology applications
======================
"""

from .mcfit import mcfit
from . import kernels
import numpy


__all__ = ['P2xi', 'xi2P', 'DoubleSphericalBessel', 'TophatVar', 'GaussVar', 'ExcursionSet']


class P2xi(mcfit):
    """Power spectrum to correlation function
    """
    def __init__(self, k, l=0, q=1.5, N=None, lowring=True):
        self.l = l
        UK = kernels.Mellin_SphericalBesselJ(l)
        mcfit.__init__(self, k, UK, q, N=N, lowring=lowring)
        phase = (-1 if l & 2 else 1) * (1j if l & 1 else 1) # i^l
        self.prefac *= phase / (2*numpy.pi)**1.5 * self.x**3


class xi2P(mcfit):
    """Correlation function to power spectrum,
    also radial profile to its Fourier transform
    """
    def __init__(self, r, l=0, q=1.5, N=None, lowring=True):
        self.l = l
        UK = kernels.Mellin_SphericalBesselJ(l)
        mcfit.__init__(self, r, UK, q, N=N, lowring=lowring)
        self.prefac *= self.x**3
        phase = (-1 if l & 2 else 1) * (1j if l & 1 else 1) # i^l
        self.postfac *= (2*numpy.pi)**1.5 / phase


from .transforms import DoubleSphericalBessel # backward compatibility


class TophatVar(mcfit):
    r"""Variance in a top-hat window

    Examples
    --------
    To compute :math:`\sigma_8` of a linear power spectrum :math:`P(k)`, with
    k in unit of :math:`h/\mathrm{Mpc}` and P in unit of :math:`\mathrm{Mpc}^3/h^3`
    >>> R, var = TophatVar(k)(P)
    >>> from scipy.interpolate import CubicSpline
    >>> varR = CubicSpline(R, var)
    >>> sigma8 = numpy.sqrt(varR(8))
    """
    def __init__(self, k, q=1.5, N=None, lowring=True):
        UK = kernels.Mellin_TophatSq(3)
        mcfit.__init__(self, k, UK, q, N=N, lowring=lowring)
        self.prefac *= self.x**3 / (2 * numpy.pi**2)


class GaussVar(mcfit):
    """Variance in a Gaussian window
    """
    def __init__(self, k, q=1.5, N=None, lowring=True):
        UK = kernels.Mellin_GaussSq()
        mcfit.__init__(self, k, UK, q, N=N, lowring=lowring)
        self.prefac *= self.x**3 / (2 * numpy.pi**2)


class ExcursionSet(mcfit):
    """Excursion set trajectory
    BCEK 91 model
    """
    def __init__(self, k, q=0, N=None, lowring=True):
        raise NotImplementedError
