from numpy import exp, logspace
from numpy.testing import assert_allclose
from fit.transforms import Hankel

def test_Hankel():
    x = logspace(-3, 3, num=60, endpoint=False)
    F = 1 / (1 + x*x)**1.5
    H = Hankel(x, nu=0, q=0.5, N=256, lowring=True)
    H.check(F)
    y, G = H(F)
    Gexact = exp(-y)
    assert_allclose(G, Gexact, rtol=1e-9, atol=1e-11)

    x = logspace(-3, 3, num=60, endpoint=False)
    F = exp(-x)
    H = Hankel(x, nu=0, q=1, N=256, lowring=True)
    H.check(F)
    y, G = H(F)
    Gexact = 1 / (1 + y*y)**1.5
    assert_allclose(G, Gexact, rtol=1e-9, atol=1e-11)
