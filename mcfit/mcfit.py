from __future__ import print_function, division
from numpy import pi, abs, ceil, exp, log, log2, angle, \
                array, arange, zeros, concatenate, searchsorted, allclose
from numpy.fft import rfft, hfft


class mcfit(object):
    r"""Multiplicatively Convolutional Fast Integral Transform

    Compute integral transforms of the form
    .. math:: G(y) = \int_0^\infty F(x) K(xy) \,\frac{\mathrm{d}x}x
    using the FFTLog [1]_[2]_ algorithm.
    Here :math:`F(x)` is the input function, :math:`G(y)` is the output
    function, and :math:`K(xy)` is the integral kernel.

    Equivalently
    .. math:: g(y) = \int_0^\infty f(x) (xy)^q K(xy) \,\frac{\mathrm{d}x}x
    where :math:`f(x)=x^{-q}F(x)`, :math:`g(y)=y^q G(y)`, and the tilt
    :math:`q` is a free parameter serving to shift power of :math:`x` between
    the input function :math:`F(x)` and the kernel.

    Parameters
    ----------
    x : float, array_like
        log-evenly spaced argument of F
    UK : callable
        Mellin transform of the kernel
        .. math:: U_K(z) \equiv \int_0^\infty t^z K(t) \, \mathrm{d}t
    q : float
        power-law tilt, can be used to balance f at large and small x.
        Avoid the singularities in UK
    N : int, optional
        length of 1D FFT, defaults to the smallest power of 2 that doubles the
        length of x
    lowring : bool, optional
        if True and N is even, set y according to the low-ringing condition,
        otherwise
        :math:`x_\mathrm{min}*y_\mathrm{max}=x_\mathrm{max}*y_\mathrm{min}=1`
    prefac : float, array_like
        multiply the input function by prefac before the transform, serving to
        convert an integral to the general form (apart from the tilt factor
        :math:`x^{-q}`)
    postfac : float, array_like
        multiply the output function by postfac after the transform, serving to
        convert an integral to the general form (apart from the tilt factor
        :math:`y^{-q}`)

    Methods
    -------
    __call__
    check

    Examples
    --------
    >>> x = numpy.logspace(-3, 3, num=60, endpoint=False)
    >>> F = 1 / (1 + x*x)**1.5
    >>> H = mcfit(x, mcfit.kernels.Mellin_BesselJ(0))
    >>> H.check(F)
    >>> y, G = H(F)
    >>> Gexact = numpy.exp(-y)
    >>> numpy.allclose(G, Gexact)

    More conveniently, use the Hankel subclass
    >>> y, G = mcfit.transforms.Hankel(x)(F)

    Notes
    -----
    Caveats about q

    References
    ----------
    .. [1] J. D. Talman. Numerical Fourier and Bessel Transforms in Logarithmic Variables.
            Journal of Computational Physics, 29:35-48, October 1978.
    .. [2] A. J. S. Hamilton. Uncorrelated modes of the non-linear power spectrum.
            MNRAS, 312:257-284, February 2000.
    """

    def __init__(self, x, UK, q, N=None, lowring=True, prefac=1, postfac=1):
        self.x = x
        self.UK = UK
        self.q = q
        self._setup(N)
        self.lowring = lowring
        self.prefac = prefac
        self.postfac = postfac


    def _setup(self, N):
        """Internal function to validate x, to set N and y, and to compute
        coefficients :math:`u_m`
        """
        Nx = len(self.x)
        if Nx < 2:
            raise ValueError("length of argument is too short")
        Delta = log(self.x[-1] / self.x[0]) / (Nx - 1)
        if not allclose(self.x[1:] / self.x[:-1], exp(Delta), rtol=1e-3):
            raise ValueError("input argument must be log-evenly spaced")

        if N is None:
            folds = int(ceil(log2(Nx))) + 1
            self.N = 2**folds
        else:
            self.N = N
        if self.N < Nx:
            raise ValueError("N is shorter than the length of argument")
        self._Npad = self.N - Nx

        lnxy = 0 # = lnxmin + lnymax = lnxmax + lnymin
        if self.lowring and self.N % 2 == 0:
            lnxy = Delta / pi * angle(self.UK(self.q + 1j * pi / Delta))
        self.y = exp(lnxy - Delta) / self.x[::-1]

        m = arange(0, self.N//2 + 1)
        self._u = self.UK(self.q + 2j * pi / self.N / Delta * m)
        self._u *= exp(-2j * pi * lnxy / self.N / Delta * m)

        # following is unnecessary, for hfft ignores the imag at Nyquist anyway
        # if not self.lowring and self.N % 2 == 0:
        #     self._u[self.N//2] = self._u[self.N//2].real


    def __call__(self, F, extrap=True):
        """Evaluate the integral

        Parameters
        ----------
        F : float, array_like
            input function, internally padded symmetrically to length N with
            power-law extrapolations or zeros
        extrap : bool or 2-tuple of bools, optional
            whether to extrapolate F with power laws or to just pad with zeros
            for the internal paddings.
            In case of a tuple, the two elements are for the left and right
            sides of F respectively

        Returns
        -------
        y : float, array_like
            log-evenly spaced argument of G
        G : float, array_like
            output function, with internal paddings discarded
        """
        if len(F) != len(self.x):
            raise ValueError("lengths of input function and argument must match")

        f = self.prefac * self.x**(-self.q) * F

        # pad with power-law extrapolations or zeros
        if isinstance(extrap, bool):
            extrap_l = extrap_r = extrap
        elif isinstance(extrap, (tuple, list)) and len(extrap) == 2 and \
                all(isinstance(e, bool) for e in extrap):
            extrap_l, extrap_r = *extrap
        else:
            raise TypeError("extrap must be either a bool or a tuple of two bools")
        if extrap_l:
            fpad_l = f[0] * (f[1] / f[0]) ** arange(-(self._Npad//2), 0)
        else:
            fpad_l = zeros(self._Npad//2)
        if extrap_r:
            fpad_r = f[-1] * (f[-1] / f[-2]) ** arange(1, self._Npad - self._Npad//2 + 1)
        else:
            fpad_r = zeros(self._Npad - self._Npad//2)
        f = concatenate((fpad_l, f, fpad_r))

        # convolution
        f = rfft(f) # f(x_n) -> f_m
        g = f * self._u # f_m -> g_m
        g = hfft(g, self.N) / self.N # g_m -> g(y_n)

        # discard paddings
        g = g[self._Npad - self._Npad//2 : self.N - self._Npad//2]

        G = self.postfac * self.y**(-self.q) * g

        return self.y, G


    def check(self, F):
        """rough sanity checks on the input function
        """
        f = self.prefac * self.x**(-self.q) * F
        fabs = abs(f)

        iQ1, iQ3 = searchsorted(fabs.cumsum(), array([0.25, 0.75]) * fabs.sum())
        assert 0 != iQ1 != iQ3 != len(self.x), "checker giving up"
        fabs_l = fabs[:iQ1].mean()
        fabs_m = fabs[iQ1:iQ3].mean()
        fabs_r = fabs[iQ3:].mean()
        if fabs_l > fabs_m:
            print("left wing seems heavy: {:.2g} vs {:.2g}, "
                    "change tilt and mind convergence".format(fabs_l, fabs_m))
        if fabs_m < fabs_r:
            print("right wing seems heavy: {:.2g} vs {:.2g}, "
                    "change tilt and mind convergence".format(fabs_m, fabs_r))

        if fabs[0] > fabs[1]:
            print("left tail may blow up: {:.2g} vs {:.2g}, "
                    "change tilt or avoid extrapolation".format(f[0], f[1]))
        if fabs[-2] < fabs[-1]:
            print("right tail may blow up: {:.2g} vs {:.2g}, "
                    "change tilt or avoid extrapolation".format(f[-2], f[-1]))

        if f[0]*f[1] < 0:
            print("left tail looks wiggly: {:.2g} vs {:.2g}, "
                    "avoid extrapolation".format(f[0], f[1]))
        if f[-2]*f[-1] < 0:
            print("right tail looks wiggly: {:.2g} vs {:.2g}, "
                    "avoid extrapolation".format(f[-2], f[-1]))
