import numpy
from numpy.testing import assert_allclose
from ..transforms import *


def test_matrix():
    N = 81
    x = numpy.logspace(-3, 3, num=N, endpoint=False)
    F = 1 / (1 + x*x)**1.5

    H1 = Hankel(x, nu=0, q=1, N=N)
    y, G = H1(F)

    a1, b1, C1 = H1.matrix(full=False)
    M1 = H1.matrix(full=True)

    assert_allclose(a1.ravel() * (C1 @ (b1 * F)), G, rtol=1e-10, atol=0)
    assert_allclose(M1 @ F, G, rtol=1e-10, atol=0)

    H2 = Hankel(y, nu=0, q=1, N=N)

    a2, b2, C2 = H2.matrix(full=False)
    M2 = H2.matrix(full=True)

    assert_allclose(C1 @ C2, numpy.eye(N), rtol=0, atol=1e-14)
    assert_allclose(C2 @ C1, numpy.eye(N), rtol=0, atol=1e-14)
    assert_allclose(M1 @ M2, numpy.eye(N), rtol=0, atol=1e-9)
    assert_allclose(M2 @ M1, numpy.eye(N), rtol=0, atol=1e-9)


def test_pad():
    x = numpy.logspace(-3, 3, num=6, endpoint=False)
    _x_ = numpy.logspace(-6, 6, num=13, endpoint=True)
    y, _y_ = 1/x[::-1], 1/_x_[::-1]

    H = Hankel(x, N=13)

    assert_allclose(H._pad(x, True, False), _x_)
    assert_allclose(H._pad(y, True, True), _y_)

    assert_allclose(H._unpad(_x_, False), x)
    assert_allclose(H._unpad(_y_, True), y)
