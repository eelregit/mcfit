from numpy import arange, exp, log, ndim, pi, sqrt
from scipy.special import gamma
try:
    from scipy.special import loggamma
except ImportError:
    def loggamma(x):
        return log(gamma(x))

def _deriv(UK, deriv):
    """Real deriv is to :math:`t`, complex deriv is to :math:`\ln t`"""
    if deriv == 0:
        return UK

    if isinstance(deriv, complex):
        def UKderiv(z):
            return (-z) ** deriv.imag * UK(z)
        return UKderiv

    def UKderiv(z):
        poly = arange(deriv) + 1
        poly = poly - z if ndim(z) == 0 else poly - z.reshape(-1, 1)
        poly = poly.prod(axis=-1)
        return poly * UK(z - deriv)
    return UKderiv

def Mellin_BesselJ(nu, deriv=0):
    def UK(z):
        return exp(log(2)*(z-1) + loggamma(0.5*(nu+z)) - loggamma(0.5*(2+nu-z)))
    return _deriv(UK, deriv)

def Mellin_SphericalBesselJ(nu, deriv=0):
    def UK(z):
        return exp(log(2)*(z-1.5) + loggamma(0.5*(nu+z)) - loggamma(0.5*(3+nu-z)))
    return _deriv(UK, deriv)

def Mellin_FourierSine(deriv=0):
    def UK(z):
        return exp(log(2)*(z-0.5) + loggamma(0.5*(1+z)) - loggamma(0.5*(2-z)))
    return _deriv(UK, deriv)

def Mellin_FourierCosine(deriv=0):
    def UK(z):
        return exp(log(2)*(z-0.5) + loggamma(0.5*z) - loggamma(0.5*(1-z)))
    return _deriv(UK, deriv)

def Mellin_DoubleBesselJ(alpha, nu1, nu2):
    import mpmath
    from numpy import frompyfunc
    hyp2f1 = frompyfunc(lambda *a: complex(mpmath.hyp2f1(*a)), 4, 1)
    if 0 < alpha < 1:
        def UK(z):
            return exp(log(2)*(z-1) + log(alpha)*nu2 + loggamma(0.5*(nu1+nu2+z))
                            - loggamma(0.5*(2+nu1-nu2-z)) - loggamma(1+nu2)) \
                    * hyp2f1(0.5*(-nu1+nu2+z), 0.5*(nu1+nu2+z), 1+nu2, alpha**2)
    elif alpha > 1:
        def UK(z):
            return exp(log(2)*(z-1) + log(alpha)*(-nu1-z) + loggamma(0.5*(nu1+nu2+z))
                            - loggamma(0.5*(2-nu1+nu2-z)) - loggamma(1+nu1)) \
                    * hyp2f1(0.5*(nu1-nu2+z), 0.5*(nu1+nu2+z), 1+nu1, alpha**-2)
    elif alpha == 1:
        def UK(z):
            return exp(log(2)*(z-1) + loggamma(1-z) + loggamma(0.5*(nu1+nu2+z))
                            - loggamma(0.5*(2+nu1-nu2-z))- loggamma(0.5*(2-nu1+nu2-z))
                            - loggamma(0.5*(2+nu1+nu2-z)))
    else:
        raise ValueError
    return UK

def Mellin_DoubleSphericalBesselJ(alpha, nu1, nu2):
    import mpmath
    from numpy import frompyfunc
    hyp2f1 = frompyfunc(lambda *a: complex(mpmath.hyp2f1(*a)), 4, 1)
    if 0 < alpha < 1:
        def UK(z):
            return pi * exp(log(2)*(z-3) + log(alpha)*nu2 + loggamma(0.5*(nu1+nu2+z))
                            - loggamma(0.5*(3+nu1-nu2-z)) - loggamma(1.5+nu2)) \
                    * hyp2f1(0.5*(-1-nu1+nu2+z), 0.5*(nu1+nu2+z), 1.5+nu2, alpha**2)
    elif alpha > 1:
        def UK(z):
            return pi * exp(log(2)*(z-3) + log(alpha)*(-nu1-z) + loggamma(0.5*(nu1+nu2+z))
                            - loggamma(0.5*(3-nu1+nu2-z)) - loggamma(1.5+nu1)) \
                    * hyp2f1(0.5*(-1+nu1-nu2+z), 0.5*(nu1+nu2+z), 1.5+nu1, alpha**-2)
    elif alpha == 1:
        def UK(z):
            return pi * exp(log(2)*(z-3) + loggamma(2-z) + loggamma(0.5*(nu1+nu2+z))
                            - loggamma(0.5*(3+nu1-nu2-z))- loggamma(0.5*(3-nu1+nu2-z))
                            - loggamma(0.5*(4+nu1+nu2-z)))
    else:
        raise ValueError
    return UK

def Mellin_Tophat(dim, deriv=0):
    def UK(z):
        return exp(log(2)*(z-1) + loggamma(1+0.5*dim) + loggamma(0.5*z) \
                - loggamma(0.5*(2+dim-z)))
    return _deriv(UK, deriv)

def Mellin_TophatSq(dim, deriv=0):
    if dim == 1:
        def UK(z):
            return -0.25*sqrt(pi) * exp(loggamma(0.5*(z-2)) - loggamma(0.5*(3-z)))
    elif dim == 3:
        def UK(z):
            return 2.25*sqrt(pi)*(z-2)/(z-6) \
                    * exp(loggamma(0.5*(z-4)) - loggamma(0.5*(5-z)))
    else:
        def UK(z):
            return exp(log(2)*(dim-1) + 2*loggamma(1+0.5*dim) \
                    + loggamma(0.5*(1+dim-z)) + loggamma(0.5*z) \
                    - loggamma(1+dim-0.5*z) - loggamma(0.5*(2+dim-z))) / sqrt(pi)
    return _deriv(UK, deriv)

def Mellin_Gauss(deriv=0):
    def UK(z):
        return 2**(0.5*z-1) * gamma(0.5*z)
    return _deriv(UK, deriv)

def Mellin_GaussSq(deriv=0):
    def UK(z):
        return 0.5 * gamma(0.5*z)
    return _deriv(UK, deriv)
