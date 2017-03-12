from numpy import exp, log, pi, sqrt
from scipy.special import loggamma, gamma

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

def Mellin_Tophat(d):
    def UK(z):
        return exp(log(2)*(z-1) + loggamma(1+0.5*d) + loggamma(0.5*z) - loggamma(0.5*(2+d-z)))
    return UK

def Mellin_TophatSq(d):
    if d == 1:
        def UK(z):
            return -0.25*sqrt(pi)  \
                    * exp(loggamma(0.5*(z-2)) - loggamma(0.5*(3-z)))
    elif d == 3:
        def UK(z):
            return 2.25*sqrt(pi)*(z-2)/(z-6)  \
                    * exp(loggamma(0.5*(z-4)) - loggamma(0.5*(5-z)))
    else:
        def UK(z):
            return exp(log(2)*(d-1) + 2*loggamma(1+0.5*d)  \
                    + loggamma(0.5*(1+d-z)) + loggamma(0.5*z)  \
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
