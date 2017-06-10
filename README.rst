mcfit: Multiplicatively Convolutional Fast Integral Transforms
============================================================

`mcfit` computes integral transforms of the form

.. math:: G(y) = \int_0^\infty F(x) K(xy) \,\frac{\mathrm{d}x}x

where :math:`K(xy)` is the integral kernel as a function of
the product of the input and output arguments.

`mcfit` implements the FFTLog algorithm.
The idea is to take advantage of the convolution theorem
in :math:`\ln x` and :math:`\ln y`.
It approximate the input function with truncated Fourier series,
and use the exact Fourier transform of the kernel (can be oscillatory).
One can calculate the latter analytically via Mellin transform.
This algorithm is optimal when the input function is smooth
and spans a large range in :math:`\ln x`.
