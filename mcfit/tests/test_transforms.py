import numpy
from numpy.testing import assert_allclose
from ..transforms import *

def test_Hankel():
    def F_fun(x): return 1 / (1 + x*x)**1.5
    def G_fun(y): return numpy.exp(-y)

    x = numpy.logspace(-3, 3, num=60, endpoint=False)
    F = F_fun(x)
    H = Hankel(x, nu=0, q=1, N=128, lowring=True)
    y, G = H(F)
    assert_allclose(G, G_fun(y), rtol=1e-8, atol=1e-8)

    # NOTE the range for best accuracy does not exactly "match"
    y = numpy.logspace(-4, 2, num=60, endpoint=False)
    G = G_fun(y)
    H_inv = Hankel(y, nu=0, q=1, N=128, lowring=True)
    x, F = H_inv(G)
    assert_allclose(F, F_fun(x), rtol=1e-10, atol=1e-10)
