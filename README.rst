mcfit: Multiplicatively Convolutional Fast Integral Transforms
==============================================================

`mcfit` computes integral transforms of the form
.. math:: G(y) = \int_0^\infty F(x) K(xy) \,\frac{\mathrm{d}x}x
where :math:`F(x)` is the input function, :math:`G(y)` is the output function,
and :math:`K(xy)` is the integral kernel.
Equivalently
.. math:: g(y) = \int_0^\infty f(x) (xy)^q K(xy) \,\frac{\mathrm{d}x}x
where :math:`f(x)=x^{-q}F(x)`, :math:`g(y)=y^q G(y)`, and the tilt :math:`q` is
a free parameter serving to shift power of :math:`x` between the input function
:math:`F(x)` and the kernel.

`mcfit` implements the FFTLog algorithm.
The idea is to take advantage of the convolution theorem in :math:`\ln x` and
:math:`\ln y`.
It approximate the input function with truncated Fourier series, and use the
exact Fourier transform of the (possibly oscillatory) kernel.
One can calculate the latter analytically via Mellin transform.
This algorithm is optimal when the input function is smooth and spans a large
range in :math:`\ln x`.
