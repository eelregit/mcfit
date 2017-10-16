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
        self.prefac *= numpy.real_if_close(1j**l) / (2*numpy.pi)**1.5 * self.x**3


class xi2P(mcfit):
    """Correlation function to power spectrum,
    also radial profile to its Fourier transform
    """
    def __init__(self, r, l=0, q=1.5, N=None, lowring=True):
        self.l = l
        UK = kernels.Mellin_SphericalBesselJ(l)
        mcfit.__init__(self, r, UK, q, N=N, lowring=lowring)
        self.prefac *= self.x**3
        self.postfac *= (2*numpy.pi)**1.5 / numpy.real_if_close(1j**l)


class DoubleSphericalBessel(mcfit):
    r"""Compute integrals with two spherical Bessel functions
    .. math:: G(y_1; \alpha) \equiv G(y_1, y_2=\alpha y_1)
                = \int_0^\infty F(x) j_{l_1}(xy_1) j_{l_2}(xy_2) \,x^2\d x

    Parameters
    ----------
    alpha : float
        y2 / y1
    l : int, optional
        default is 0
    l1 : int, optional
        default is l
    l2 : int, optional
        default is l
    """
    def __init__(self, x, alpha, l=0, l1=None, l2=None, q=1.5, N=None, lowring=True):
        self.alpha = alpha
        if l1 is None:
            l1 = l
        if l2 is None:
            l2 = l
        self.l1 = l1
        self.l2 = l2
        UK = kernels.Mellin_DoubleSphericalBesselJ(alpha, l1, l2)
        mcfit.__init__(self, x, UK, q, N=N, lowring=lowring)
        self.prefac *= self.x**3


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
