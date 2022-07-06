"""Cosmology applications"""

from mcfit.mcfit import mcfit
from mcfit import kernels
from numpy import pi


__all__ = ['P2xi', 'xi2P', 'TophatVar', 'GaussVar']


class P2xi(mcfit):
    """Power spectrum to correlation function.

    Parameters
    ----------
    k : see `x` in :class:`mcfit.mcfit`
    l : int
        order
    n : int
        to generalize correlation function with extra power law factor
        :math:`k^n` in the integrand. If not None, the phase factor is ignored.
        The tilt parameter `q` is automatically adjusted (to `q+n`) based on
        the provided value

    See :class:`mcfit.mcfit`
    """
    def __init__(self, k, l=0, n=None, deriv=0, q=1.5, **kwargs):
        self.l = l
        MK = kernels.Mellin_SphericalBesselJ(l, deriv)

        if n is None:
            phase = (-1 if l & 2 else 1) * (1j if l & 1 else 1)  # i^l
            n = 0
        else:
            phase = 1

        mcfit.__init__(self, k, MK, q+n, **kwargs)
        self.prefac *= self.x**(3+n) / (2*pi)**1.5
        self.postfac *= phase


class xi2P(mcfit):
    """Correlation function to power spectrum, also radial profile to its
    Fourier transform.

    Parameters
    ----------
    r : see `x` in :class:`mcfit.mcfit`
    l : int
        order

    See :class:`mcfit.mcfit`
    """
    def __init__(self, r, l=0, deriv=0, q=1.5, **kwargs):
        self.l = l
        MK = kernels.Mellin_SphericalBesselJ(l, deriv)
        mcfit.__init__(self, r, MK, q, **kwargs)
        self.prefac *= self.x**3
        phase = (-1 if l & 2 else 1) * (1j if l & 1 else 1)  # i^l
        self.postfac *= (2*pi)**1.5 / phase


class TophatVar(mcfit):
    r"""Variance in a top-hat window.

    Parameters
    ----------
    k : see `x` in :class:`mcfit.mcfit`

    Examples
    --------
    To compute :math:`\sigma_8` of a linear power spectrum :math:`P(k)`
    >>> R, var = TophatVar(k, lowring=True)(P, extrap=True)
    >>> from scipy.interpolate import CubicSpline
    >>> varR = CubicSpline(R, var)
    >>> sigma8 = numpy.sqrt(varR(8))

    See :class:`mcfit.mcfit`
    """
    def __init__(self, k, deriv=0, q=1.5, **kwargs):
        MK = kernels.Mellin_TophatSq(3, deriv)
        mcfit.__init__(self, k, MK, q, **kwargs)
        self.prefac *= self.x**3 / (2 * pi**2)


class GaussVar(mcfit):
    """Variance in a Gaussian window.

    Parameters
    ----------
    k : see `x` in :class:`mcfit.mcfit`

    See :class:`mcfit.mcfit`
    """
    def __init__(self, k, deriv=0, q=1.5, **kwargs):
        MK = kernels.Mellin_GaussSq(deriv)
        mcfit.__init__(self, k, MK, q, **kwargs)
        self.prefac *= self.x**3 / (2 * pi**2)
