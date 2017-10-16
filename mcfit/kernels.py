from numpy import exp, log, pi, sqrt
from scipy.special import gamma
try:
    from scipy.special import loggamma
except ImportError:
    def loggamma(x):
        return log(gamma(x))

def Mellin_BesselJ(nu):
    def UK(z):
        return exp(log(2)*(z-1) + loggamma(0.5*(nu+z)) - loggamma(0.5*(2+nu-z)))
    return UK

def Mellin_SphericalBesselJ(nu):
    def UK(z):
        return exp(log(2)*(z-1.5) + loggamma(0.5*(nu+z)) - loggamma(0.5*(3+nu-z)))
    return UK

def Mellin_FourierSine():
    def UK(z):
        return exp(log(2)*(z-0.5) + loggamma(0.5*(1+z)) - loggamma(0.5*(2-z)))
    return UK

def Mellin_FourierCosine():
    def UK(z):
        return exp(log(2)*(z-0.5) + loggamma(0.5*z) - loggamma(0.5*(1-z)))
    return UK

def Mellin_DoubleSphericalBesselJ(alpha, l1, l2):
    import mpmath
    from numpy import frompyfunc
    hyp2f1 = frompyfunc(lambda *a: complex(mpmath.hyp2f1(*a)), 4, 1)
    if 0 < alpha <= 1:
        def UK(z):
            return pi * exp(log(2)*(z-3) + log(alpha)*l2 + loggamma(0.5*(l1+l2+z))
                            - loggamma(0.5*(3+l1-l2-z)) - loggamma(1.5+l2)) \
                    * hyp2f1(0.5*(-1-l1+l2+z), 0.5*(l1+l2+z), 1.5+l2, alpha**2)
    elif alpha > 1:
        def UK(z):
            return pi * exp(log(2)*(z-3) + log(alpha)*(-l1-z) + loggamma(0.5*(l1+l2+z))
                            - loggamma(0.5*(3-l1+l2-z)) - loggamma(1.5+l1)) \
                    * hyp2f1(0.5*(-1+l1-l2+z), 0.5*(l1+l2+z), 1.5+l1, alpha**-2)
    else:
        raise ValueError
    return UK

def Mellin_Tophat(d):
    def UK(z):
        return exp(log(2)*(z-1) + loggamma(1+0.5*d) + loggamma(0.5*z) \
                - loggamma(0.5*(2+d-z)))
    return UK

def Mellin_TophatSq(d):
    if d == 1:
        def UK(z):
            return -0.25*sqrt(pi) * exp(loggamma(0.5*(z-2)) - loggamma(0.5*(3-z)))
    elif d == 3:
        def UK(z):
            return 2.25*sqrt(pi)*(z-2)/(z-6) \
                    * exp(loggamma(0.5*(z-4)) - loggamma(0.5*(5-z)))
    else:
        def UK(z):
            return exp(log(2)*(d-1) + 2*loggamma(1+0.5*d) \
                    + loggamma(0.5*(1+d-z)) + loggamma(0.5*z) \
                    - loggamma(1+d-0.5*z) - loggamma(0.5*(2+d-z))) / sqrt(pi)
    return UK

def Mellin_Gauss():
    def UK(z):
        return 2**(0.5*z-1) * gamma(0.5*z)
    return UK

def Mellin_GaussSq():
    def UK(z):
        return 0.5 * gamma(0.5*z)
    return UK
