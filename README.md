# Multiplicatively Convolutional Fast Integral Transforms


## Features

* Compute integral transforms:
  $$G(y) = \int_0^\infty F(x) K(xy) \frac{dx}x;$$
* Inverse transform without analytic inversion;
* Integral kernels as derivatives:
  $$G(y) = \int_0^\infty F(x) K'(xy) \frac{dx}x;$$
* Transform input array along any axis of `ndarray`;
* Output the matrix form;
* 1-to-n transform for multiple kernels (TODO):
  $$G(y_1, \cdots, y_n) = \int_0^\infty \frac{dx}x F(x) \prod_{a=1}^n K_a(xy_a);$$
* Easily extensible for other kernels;
* Support NumPy and JAX.


## Algorithm

`mcfit` computes integral transforms of the form
  $$G(y) = \int_0^\infty F(x) K(xy) \frac{dx}x$$
where $F(x)$ is the input function, $G(y)$ is the output function, and
$K(xy)$ is the integral kernel.
One is free to scale all three functions by a power law
  $$g(y) = \int_0^\infty f(x) k(xy) \frac{dx}x$$
where $f(x)=x^{-q} F(x)$, $g(y)=y^q G(y)$, and $k(t)=t^q K(t)$.
And $q$ is a tilt parameter serving to shift power of $x$ between the
input function and the kernel.

`mcfit` implements the FFTLog algorithm.
The idea is to take advantage of the convolution theorem in $\ln x$ and
$\ln y$.
It approximates the input function with a partial sum of the Fourier
series over one period of a periodic approximant, and use the exact
Fourier transform of the kernel.
One can calculate the latter analytically as a Mellin transform.
This algorithm is optimal when the input function is smooth in $\ln x$,
and is ideal for oscillatory kernels with input spanning a wide range in
$\ln x$.


## Installation

```sh
  pip install mcfit
```


## Documentation

See docstring of `mcfit.mcfit`, which also applies to other
subclasses of transformations.
Also see `doc/mcfit.tex` for more explanations.


## Examples

One can perform the following pair of Hankel transforms
  $$e^{-y} = \int_0^\infty (1+x^2)^{-\frac32} J_0(xy) x dx, \quad (1+y^2)^{-\frac32} = \int_0^\infty e^{-x} J_0(xy) x dx$$
easily as follows
```python
  def F_fun(x): return 1 / (1 + x*x)**1.5
  def G_fun(y): return numpy.exp(-y)

  from mcfit import Hankel

  x = numpy.logspace(-3, 3, num=60, endpoint=False)
  F = F_fun(x)
  H = Hankel(x, lowring=True)
  y, G = H(F, extrap=True)
  numpy.allclose(G, G_fun(y), rtol=1e-8, atol=1e-8)

  y = numpy.logspace(-4, 2, num=60, endpoint=False)
  G = G_fun(y)
  H_inv = Hankel(y, lowring=True)
  x, F = H_inv(G, extrap=True)
  numpy.allclose(F, F_fun(x), rtol=1e-10, atol=1e-10)
```

Cosmologists often need to transform a power spectrum to its correlation
function
```python
  from mcfit import P2xi
  k, P = numpy.loadtxt('P.txt', unpack=True)
  r, xi = P2xi(k)(P)
```
and the other way around
```python
  from mcfit import xi2P
  r, xi = numpy.loadtxt('xi.txt', unpack=True)
  k, P = xi2P(r)(xi)
```

Similarly for the quadrupoles
```python
  k, P2 = numpy.loadtxt('P2.txt', unpack=True)
  r, xi2 = P2xi(k, l=2)(P2)
```

Also useful to the cosmologists is the tool below that computes the
variance of the overdensity field as a function of radius, from which
$\sigma_8$ can be interpolated.
```python
  from mcfit import TophatVar
  R, var = TophatVar(k, lowring=True)(P, extrap=True)
  from scipy.interpolate import CubicSpline
  varR = CubicSpline(R, var)
  sigma8 = numpy.sqrt(varR(8))
```
