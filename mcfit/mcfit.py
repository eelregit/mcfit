from __future__ import print_function, division
import numpy


class mcfit(object):
    r"""Compute integral transforms as a multiplicative convolution.

    The generic form is
    .. math:: G(y) = \int_0^\infty F(x) K(xy) \,\frac{\mathrm{d}x}x

    Here :math:`F(x)` is the input function, :math:`G(y)` is the output
    function, and :math:`K(xy)` is the integral kernel.
    One is free to scale all three functions by a power law

    .. math:: g(y) = \int_0^\infty f(x) k(xy) \,\frac{\mathrm{d}x}x

    in which :math:`f(x) = x^{-q} F(x)`, :math:`g(y) = y^q G(y)`, and
    :math:`k(t) = t^q K(t)`.
    The tilt parameter :math:`q` shifts power of :math:`x` between the input
    function and the kernel.

    Parameters
    ----------
    x : (Nin,) array_like
        log-evenly spaced input argument
    UK : callable
        Mellin transform of the kernel
        .. math:: U_K(z) \equiv \int_0^\infty t^z K(t) \, \mathrm{d}t
    q : float
        power-law tilt, can be used to balance f at large and small x.
        Avoid the singularities in UK
    N : int, optional
        length of FFT, defaults to the smallest power of 2 that doubles the
        length of x
    lowring : bool, optional
        if True and N is even, set y according to the low-ringing condition,
        otherwise
        :math:`x_\mathrm{min} * y_\mathrm{max} = x_\mathrm{max} * y_\mathrm{min} = 1`

    Attributes
    ----------
    Nin : int
        input length
    x : (Nin,) ndarray
        log-evenly spaced input argument
    y : (Nin,) ndarray
        log-evenly spaced output argument
    _x_ : (N,) ndarray
        padded input argument
    _y_ : (N,) ndarray
        padded output argument
    prefac : array_like
        a function of x (excluding the tilt factor :math:`x^{-q}` to be added
        automatically) to multiply before the convolution.
        Set it to convert the integral to the generic form
    postfac : array_like
        a function of y (excluding the tilt factor :math:`y^{-q}` to be added
        automatically) to multiply after the convolution.
        Set it to convert the integral to the generic form

    Methods
    -------
    __call__
    matrix
    check

    Examples
    --------
    >>> x = numpy.logspace(-3, 3, num=60, endpoint=False)
    >>> F = 1 / (1 + x*x)**1.5
    >>> H = mcfit.mcfit(x, mcfit.kernels.Mellin_BesselJ(0), q=1)
    >>> y, G = H(x**2 * F) # x^2 factor to reduce it to the generic form
    >>> Gexact = numpy.exp(-y)
    >>> numpy.allclose(G, Gexact)

    More conveniently, use the Hankel transform subclass
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

    def __init__(self, x, UK, q, N=None, lowring=True, prefac=None, postfac=None):
        self.x = numpy.asarray(x)
        self.Nin = len(x)
        self.UK = UK
        self.q = q
        self.N = N
        self.lowring = lowring
        self._setup()
        self.prefac = 1 if prefac is None else prefac
        self.postfac = 1 if postfac is None else postfac
        if prefac is not None or postfac is not None:
            import warnings
            msg = "prefac and postfac as parameters are deprecated. " \
            "Use them as attributes instead. See cosmology.xi2P.__init__ " \
            "for an example. This gives the flexibility that postfac can " \
            "be made a function of the output argument."
            warnings.warn(msg, FutureWarning)


    @property
    def prefac(self):
        return self._prefac

    @prefac.setter
    def prefac(self, value):
        self._prefac = value
        self._xfac = self._prefac * self.x**(-self.q)

    @property
    def postfac(self):
        return self._postfac

    @postfac.setter
    def postfac(self, value):
        self._postfac = value
        self._yfac = self._postfac * self.y**(-self.q)

    @property
    def _x_(self):
        if not hasattr(self, "_x"):
            self._x = self._pad(self.x, True, False)
        return self._x

    @property
    def _y_(self):
        if not hasattr(self, "_y"):
            self._y = self._pad(self.y, True, True)
        return self._y


    def _setup(self):
        """Validate x, set N and y, and compute :math:`u_m`.
        """
        if self.Nin < 2:
            raise ValueError("input length too short")
        Delta = numpy.log(self.x[-1] / self.x[0]) / (self.Nin - 1)
        if not numpy.allclose(self.x[1:] / self.x[:-1], numpy.exp(Delta), rtol=1e-3):
            raise ValueError("input not log-evenly spaced")

        if self.N is None:
            folds = int(numpy.ceil(numpy.log2(self.Nin))) + 1
            self.N = 2**folds
        if self.N < self.Nin:
            raise ValueError("total length shorter than input length")

        lnxy = 0 # = lnxmin + lnymax = lnxmax + lnymin
        if self.lowring and self.N % 2 == 0:
            lnxy = Delta / numpy.pi * numpy.angle(self.UK(self.q + 1j * numpy.pi / Delta))
        self.y = numpy.exp(lnxy - Delta) / self.x[::-1]

        m = numpy.arange(0, self.N//2 + 1)
        self._u = self.UK(self.q + 2j * numpy.pi / self.N / Delta * m)
        self._u *= numpy.exp(-2j * numpy.pi * lnxy / self.N / Delta * m)

        # following is unnecessary because hfft ignores the imag at Nyquist anyway
        #if not self.lowring and self.N % 2 == 0:
        #    self._u[self.N//2] = self._u[self.N//2].real


    def __call__(self, F, extrap=True, interp=False):
        """Evaluate the integral.

        Parameters
        ----------
        F : (Nin,) array_like
            input function, internally padded symmetrically to length N with
            power-law extrapolations or zeros
        extrap : bool or 2-tuple of bools, optional
            whether to extrapolate F with power laws or to just pad with zeros;
            for a tuple, the two elements are for the left and right pads
        interp : bool, optional
            when True return an interpolant computed using padded input,
            otherwise return unpadded arrays

        Returns
        -------
        y : (Nin,) ndarray
            log-evenly spaced output argument, unpadded
        G : (Nin,) ndarray
            output function, unpadded
        Gy : scipy.interpolate.CubicSpline
            output interpolant computed using padded input
        """
        if len(F) != self.Nin:
            raise ValueError("lengths of input function and argument must match")

        f = self._xfac * F
        f = self._pad(f, extrap, False)

        # convolution
        f = numpy.fft.rfft(f) # f(x_n) -> f_m
        g = f * self._u # f_m -> g_m
        g = numpy.fft.hfft(g, n=self.N) / self.N # g_m -> g(y_n)

        if not interp:
            g = self._unpad(g, True)
            G = self._yfac * g
            return self.y, G
        else:
            _G_ = self._pad(self._yfac, True, True) * g
            from scipy.interpolate import CubicSpline
            Gy = CubicSpline(self._y_, _G_)
            return Gy


    def matrix(self, full=False):
        """Return matrix form of the integral transform.

        Parameters
        ----------
        full : bool, optional
            when False return two vector factors and convolution matrix
            separately, otherwise return full transformation matrix

        Returns
        -------
        If full is False, output separately
        a : (1, N) ndarray
            "After" factor, function of y including the `postfac` and the
            power-law tilt
        b : (N,) ndarray
            "Before" factor, function of x including the `prefac` and the
            power-law tilt
        C : (N, N) ndarray
            Convolution matrix, circulant

        Otherwise, output the full matrix, combining a, b, and C
        M : (N, N) ndarray
            Full transformation matrix, `M = a * C * b`

        Notes
        -----
        M, a, b, and C are padded.

        This is not meant for evaluation with matrix multiplication but in case
        one is interested in the tranformation itself.

        When N is even and lowing is False, :math:`C C^{-1}` and :math:`M
        M^{-1}` can deviate from the identity matrix because the imaginary part
        of the Nyquist modes are dropped.

        The convolution matrix is a circulant matrix, with its first row and
        first column being the Fourier transform of :math:`u_m`.
        Indeed :math:`u_m` are the eigenvalues of the convolution matrix, that
        are diagonalized by the DFT matrix.
        Thus :math:`1/u_m` are the eigenvalues of the inverse convolution
        matrix.
        """
        a = self._pad(self._yfac, True, True)[:, numpy.newaxis]
        b = self._pad(self._xfac, True, False)
        v = numpy.fft.hfft(self._u, n=self.N) / self.N
        idx = sum(numpy.ogrid[0:self.N, -self.N:0])
        C = v[idx] # follow scipy.linalg.{circulant,toeplitz,hankel}
        if not full:
            return a, b, C
        else:
            return a * C * b


    def _pad(self, a, extrap, out):
        """Pad an array with power-law extrapolations or zeros.

        Parameters
        ----------
        a : (Nin,) ndarray
            array to be padded to length N
        extrap : bool or 2-tuple of bools
            whether to extrapolate a with power laws or to just pad with zeros;
            for a tuple, the two elements are for the left and right pads
        out : bool
            pad the input when False, otherwise the output;
            the two cases have their left and right pad sizes reversed
        """
        assert len(a) == self.Nin

        if isinstance(extrap, bool):
            _extrap = extrap_ = extrap
        elif isinstance(extrap, tuple) and len(extrap) == 2 and \
                all(isinstance(e, bool) for e in extrap):
            _extrap, extrap_ = extrap
        else:
            raise TypeError("extrap neither one bool nor a tuple of two")

        Npad = self.N - self.Nin
        if out:
            _Npad, Npad_ = Npad - Npad//2, Npad//2
        else:
            _Npad, Npad_ = Npad//2, Npad - Npad//2
        if _extrap:
            _a = a[0] * (a[1] / a[0]) ** numpy.arange(-_Npad, 0)
        else:
            _a = numpy.zeros(_Npad)
        if extrap_:
            a_ = a[-1] * (a[-1] / a[-2]) ** numpy.arange(1, Npad_ + 1)
        else:
            a_ = numpy.zeros(Npad_)

        return numpy.concatenate((_a, a, a_))


    def _unpad(self, a, out):
        """Undo padding in an array.

        Parameters
        ----------
        a : (N,) ndarray
            array to be trimmed to length Nin
        out : bool
            trim the output when True, otherwise the input;
            the two cases have their left and right pad sizes reversed
        """
        assert len(a) == self.N

        Npad = self.N - self.Nin
        if out:
            _Npad, Npad_ = Npad - Npad//2, Npad//2
        else:
            _Npad, Npad_ = Npad//2, Npad - Npad//2

        return a[_Npad : self.N - Npad_]


    def check(self, F):
        """Rough sanity checks on the input function.
        """
        f = self._xfac * F
        fabs = numpy.abs(f)

        iQ1, iQ3 = numpy.searchsorted(fabs.cumsum(), numpy.array([0.25, 0.75]) * fabs.sum())
        assert 0 != iQ1 != iQ3 != self.Nin, "checker giving up"
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

        if f[0]*f[1] <= 0:
            print("left tail looks wiggly: {:.2g} vs {:.2g}, "
                    "avoid extrapolation".format(f[0], f[1]))
        if f[-2]*f[-1] <= 0:
            print("right tail looks wiggly: {:.2g} vs {:.2g}, "
                    "avoid extrapolation".format(f[-2], f[-1]))
