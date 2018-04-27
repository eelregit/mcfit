Multiplicatively Convolutional Fast Integral Transforms
=======================================================

`mcfit` computes integral transforms of the form

.. math:: G(y) = \int_0^\infty F(x) K(xy) \frac{dx}x

where :math:`F(x)` is the input function, :math:`G(y)` is the output function,
and :math:`K(xy)` is the integral kernel.
One is free to scale all three functions by a power law

.. math:: g(y) = \int_0^\infty f(x) k(xy) \frac{dx}x

where :math:`f(x)=x^{-q}F(x)`, :math:`g(y)=y^q G(y)`, and :math:`k(t)=t^q K(t)`.
And :math:`q` is a tilt parameter serving to shift power of :math:`x` between
the input function and the kernel.

`mcfit` implements the FFTLog algorithm.
The idea is to take advantage of the convolution theorem in :math:`\ln x` and
:math:`\ln y`.
It approximates the input function with truncated Fourier series over one
period of the periodic approximant, and use the exact Fourier transform of the
possibly oscillatory kernel.
One can calculate the latter analytically as a Mellin transform.
This algorithm is optimal when the input function is smooth and spans a large
range in :math:`\ln x`.


Examples
--------

One can perform the following pair of Hankel transforms

.. math::

    e^{-y} &= \int_0^\infty (1+x^2)^{-\frac32} J_0(xy) x dx \\
    (1+y^2)^{-\frac32} &= \int_0^\infty e^{-y} J_0(xy) x dx

easily as follows

.. code-block:: python

    from mcfit import Hankel

    x = numpy.logspace(-3, 3, num=60, endpoint=False)
    F = 1 / (1 + x*x)**1.5
    H0 = Hankel(x, nu=0, q=1, N=128, lowring=True)
    y, G = H0(F)
    Gexact = numpy.exp(-y)
    numpy.allclose(G, Gexact, rtol=1e-8, atol=1e-8)

    y = numpy.logspace(-4, 2, num=60, endpoint=False)
    G = numpy.exp(-y)
    H1 = Hankel(y, nu=0, q=1, N=128, lowring=True)
    x, F = H1(G)
    Fexact = 1 / (1 + x*x)**1.5
    numpy.allclose(F, Fexact, rtol=1e-10, atol=1e-10)

Cosmologists often need to transform a power spectrum to its correlation
function

.. code-block:: python

    from mcfit import P2xi
    k, P = numpy.loadtxt('P.txt', unpack=True)
    r, xi = P2xi(k)(P)

and the other way around

.. code-block:: python

    from mcfit import xi2P
    r, xi = numpy.loadtxt('xi.txt', unpack=True)
    k, P = xi2P(r)(xi)

Similarly for the quadrupoles

.. code-block:: python

    k, P2 = numpy.loadtxt('P2.txt', unpack=True)
    r, xi2 = P2xi(k, l=2)(P2)

Also useful to the cosmologists is the tool below that computes the variance of
the overdensity field as a function of radius, from which :math:`\sigma_8` can
be interpolated.

.. code-block:: python

    R, var = TophatVar(k)(P)
    from scipy.interpolate import CubicSpline
    varR = CubicSpline(R, var)
    sigma8 = numpy.sqrt(varR(8))
