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
    # _x_, _y_, extrap pad, and unpad
    x = numpy.logspace(-3, 3, num=6, endpoint=False)
    _x_ = numpy.logspace(-6, 6, num=13, endpoint=True)
    y = numpy.logspace(-3, 3, num=6, endpoint=False)
    _y_ = numpy.logspace(-7, 5, num=13, endpoint=True)

    H = Hankel(x, N=13, xy=1)

    assert_allclose(H._x_, _x_)
    assert_allclose(H._y_, _y_)

    assert_allclose(H._pad(x, 0, True, False), _x_)
    assert_allclose(H._pad(y, 0, True, True), _y_)

    assert_allclose(H._unpad(_x_, 0, False), x)
    assert_allclose(H._unpad(_y_, 0, True), y)

    # zero pad and axis
    a = b = numpy.ones((6, 6))
    _a_ = numpy.zeros((13, 6))
    _a_[3:9, :] = 1
    _b_ = numpy.zeros((6, 13))
    _b_[:, 4:10] = 1

    assert_allclose(H._pad(a, 0, False, False), _a_)
    assert_allclose(H._pad(b, 1, False, True), _b_)
