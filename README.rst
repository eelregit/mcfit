Multiplicatively Convolutional Fast Integral Transforms
=======================================================

`mcfit` computes integral transforms of the form

.. math:: G(y) = \int_0^\infty F(x) K(xy) \,\frac{\mathrm{d}x}x

where :math:`F(x)` is the input function, :math:`G(y)` is the output function,
and :math:`K(xy)` is the integral kernel.
One is free to scale all three functions by a power law

.. math:: g(y) = \int_0^\infty f(x) k(xy) \,\frac{\mathrm{d}x}x

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
