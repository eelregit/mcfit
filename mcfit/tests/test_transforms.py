from numpy import exp, logspace
from numpy.testing import assert_allclose
from mcfit.transforms import Hankel

def test_Hankel():
    x = logspace(-3, 3, num=60, endpoint=False)
    F = 1 / (1 + x*x)**1.5
    H0 = Hankel(x, nu=0, q=1, N=128, lowring=True)
    H0.check(F)
    y, G = H0(F)
    Gexact = exp(-y)
    assert_allclose(G, Gexact, rtol=1e-8, atol=1e-8)

    # NOTE the range for best accuracy does not exactly "match"
    y = logspace(-4, 2, num=60, endpoint=False)
    G = exp(-y)
    H1 = Hankel(y, nu=0, q=1, N=128, lowring=True)
    H1.check(G)
    x, F = H1(G)
    Fexact = 1 / (1 + x*x)**1.5
    assert_allclose(F, Fexact, rtol=1e-10, atol=1e-10)
