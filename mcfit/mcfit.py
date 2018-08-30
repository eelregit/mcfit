from __future__ import division
import numpy


class mcfit(object):
    r"""Compute integral transforms as a multiplicative convolution.

    The generic form is
    .. math:: G(y) = \int_0^\infty F(x) K(xy) \frac{dx}x

    Here :math:`F(x)` is the input function, :math:`G(y)` is the output
    function, and :math:`K(xy)` is the integral kernel.
    One is free to scale all three functions by a power law

    .. math:: g(y) = \int_0^\infty f(x) k(xy) \frac{dx}x

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
        .. math:: U_K(z) \equiv \int_0^\infty t^z K(t) dt
    q : float
        power-law tilt, can be used to balance :math:`f` at large and small
        :math:`x`. Avoid the singularities in `UK`
    N : int, optional
        length of FFT, defaults to the smallest power of 2 that doubles the
        length of `x`; the input function is padded symmetrically to this
        length with extrapolations or zeros before convolution
    lowring : bool, optional
        if True and `N` is even, set `y` according to the low-ringing
        condition, otherwise see `xy`
    xy : float, optional
        reciprocal product :math:`x_{min} y_{max} = x_{max} y_{min}` when
        `lowring` is False or `N` is odd

    Attributes
    ----------
    Nin : int
        input (and output) length
    x : (Nin,) ndarray
        input argument
    y : (Nin,) ndarray
        output argument
    _x_ : (N,) ndarray
        padded `x`
    _y_ : (N,) ndarray
        padded `y`
    prefac : array_like
        a function of `x` (excluding the tilt factor :math:`x^{-q}`) to
        convert an integral to the normal form
    postfac : array_like
        a function of `y` (excluding the tilt factor :math:`y^{-q}`) to
        convert an integral to the normal form
    xfac : (Nin,) ndarray
        a function of `x` (including the tilt factor :math:`x^{-q}`) to
        multiply before the convolution
    yfac : (Nin,) ndarray
        a function of `y` (including the tilt factor :math:`y^{-q}`) to
        multiply after the convolution
    _xfac_ : (N,) ndarray
        padded `_xfac_`
    _yfac_ : (N,) ndarray
        padded `_yfac_`

    Methods
    -------
    __call__
    matrix
    check

    Examples
    --------
    >>> x = numpy.logspace(-3, 3, num=60, endpoint=False)
    >>> A = 1 / (1 + x*x)**1.5
    >>> H = mcfit.mcfit(x, mcfit.kernels.Mellin_BesselJ(0), q=1)
    >>> y, B = H(x**2 * A)
    >>> numpy.allclose(B, numpy.exp(-y))

    More conveniently, use the Hankel transform subclass
    >>> y, B = mcfit.transforms.Hankel(x)(A)

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

    def __init__(self, x, UK, q, N=None, lowring=True, xy=1):
        self.x = numpy.asarray(x)
        self.Nin = len(x)
        self.UK = UK
        self.q = q
        self.N = N
        self.lowring = lowring
        self.xy = xy

        self._setup()
        self.prefac = 1
        self.postfac = 1


    @property
    def prefac(self):
        return self._prefac

    @prefac.setter
    def prefac(self, value):
        self._prefac = value
        self.xfac = self._prefac * self.x**(-self.q)
        self._xfac_ = self._pad(self.xfac, 0, True, False)

    @property
    def postfac(self):
        return self._postfac

    @postfac.setter
    def postfac(self, value):
        self._postfac = value
        self.yfac = self._postfac * self.y**(-self.q)
        self._yfac_ = self._pad(self.yfac, 0, True, True)


    def _setup(self):
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

        lnxy = numpy.log(self.xy)
        if self.lowring and self.N % 2 == 0:
            lnxy = Delta / numpy.pi * numpy.angle(self.UK(self.q + 1j * numpy.pi / Delta))
        self.y = numpy.exp(lnxy - Delta) / self.x[::-1]

        self._x_ = self._pad(self.x, 0, True, False)
        self._y_ = self._pad(self.y, 0, True, True)

        m = numpy.arange(0, self.N//2 + 1)
        self._u = self.UK(self.q + 2j * numpy.pi / self.N / Delta * m)
        self._u *= numpy.exp(-2j * numpy.pi * lnxy / self.N / Delta * m)

        # following is unnecessary because hfft ignores the imag at Nyquist anyway
        #if not self.lowring and self.N % 2 == 0:
        #    self._u[self.N//2] = self._u[self.N//2].real


    def __call__(self, F, axis=-1, extrap=True, keeppads=False):
        """Evaluate the integral.

        Parameters
        ----------
        F : (..., Nin, ...) array_like
            input function
        axis : int, optional
            axis along which to integrate
        extrap : bool or 2-tuple of bools, optional
            whether to extrapolate `F` with power laws or to pad it with zeros,
            to length `N`, before convolution; for a tuple, the two elements
            are for the left and right pads
        keeppads : bool, optional
            whether to keep the padding in the output

        Returns
        -------
        y : (Nin,) or (N,) ndarray
            log-evenly spaced output argument
        G : (..., Nin, ...) or (..., N, ...) ndarray
            output function

        Notes
        -----
        `y`, and `G` are unpadded by default. Pads can be kept if `keeppads` is
        set to True.
        """

        F = numpy.asarray(F)

        to_axis = [1] * F.ndim
        to_axis[axis] = -1

        f = self._xfac_.reshape(to_axis) * self._pad(F, axis, extrap, False)

        # convolution
        f = numpy.fft.rfft(f, axis=axis) # f(x_n) -> f_m
        g = f * self._u.reshape(to_axis) # f_m -> g_m
        g = numpy.fft.hfft(g, n=self.N, axis=axis) / self.N # g_m -> g(y_n)

        if not keeppads:
            G = self.yfac.reshape(to_axis) * self._unpad(g, axis, True)
            return self.y, G
        else:
            _G_ = self._yfac_.reshape(to_axis) * g
            return self._y_, _G_


    def matrix(self, full=False, keeppads=True):
        """Return matrix form of the integral transform.

        Parameters
        ----------
        full : bool, optional
            when False return two vector factors and convolution matrix
            separately, otherwise return full transformation matrix
        keeppads : bool, optional
            whether to keep the padding in the output

        Returns
        -------
        If full is False, output separately
        a : (1, N) or (1, Nin) ndarray
            "After" factor, `_yfac_` or `yfac`
        b : (N,) or (Nin,) ndarray
            "Before" factor, `_xfac_` or `xfac`
        C : (N, N) or (Nin, Nin) ndarray
            Convolution matrix, circulant

        Otherwise, output the full matrix, combining `a`, `b`, and `C`
        M : (N, N) or (Nin, Nin) ndarray
            Full transformation matrix, `M = a * C * b`

        Notes
        -----
        `M`, `a`, `b`, and `C` are padded by default. Pads can be discarded if
        `keeppads` is set to False.

        This is not meant for evaluation with matrix multiplication but in case
        one is interested in the tranformation itself.

        When `N` is even and `lowring` is False, :math:`C C^{-1}` and :math:`M
        M^{-1}` can deviate from the identity matrix because the imaginary part
        of the Nyquist modes are dropped.

        The convolution matrix is a circulant matrix, with its first row and
        first column being the Fourier transform of :math:`u_m`.
        Indeed :math:`u_m` are the eigenvalues of the convolution matrix, that
        are diagonalized by the DFT matrix.
        Thus :math:`1/u_m` are the eigenvalues of the inverse convolution
        matrix.
        """

        v = numpy.fft.hfft(self._u, n=self.N) / self.N
        idx = sum(numpy.ogrid[0:self.N, -self.N:0])
        C = v[idx] # follow scipy.linalg.{circulant,toeplitz,hankel}

        if keeppads:
            a = self._yfac_.copy()
            b = self._xfac_.copy()
        else:
            a = self.yfac.copy()
            b = self.xfac.copy()
            C = self._unpad(C, 0, True)
            C = self._unpad(C, 1, False)
        a = a.reshape(-1, 1)

        if not full:
            return a, b, C
        else:
            return a * C * b


    def _pad(self, a, axis, extrap, out):
        """Pad an array with power-law extrapolations or zeros.

        Parameters
        ----------
        a : (..., Nin, ...) ndarray
            array to be padded to length `N`
        axis : int
            axis along which to pad
        extrap : bool or 2-tuple of bools
            whether to extrapolate `a` with power laws or to just pad with
            zeros; for a tuple, the two elements are for the left and right
            pads
        out : bool
            pad the output if True, otherwise the input; the two cases have
            their left and right pad sizes reversed
        """

        if axis == -1: axis += a.ndim # to fix the indexing below with axis+1

        assert a.shape[axis] == self.Nin

        to_axis = [1] * a.ndim
        to_axis[axis] = -1

        _extrap, extrap_ = extrap, extrap if numpy.isscalar(extrap) else extrap

        Npad = self.N - self.Nin
        if out:
            _Npad, Npad_ = Npad - Npad//2, Npad//2
        else:
            _Npad, Npad_ = Npad//2, Npad - Npad//2
        if _extrap:
            start = numpy.take(a, [0], axis=axis)
            ratio = numpy.take(a, [1], axis=axis) / start
            exp = numpy.arange(-_Npad, 0).reshape(to_axis)
            _a = start * ratio ** exp
        else:
            _a = numpy.zeros(a.shape[:axis] + (_Npad,) + a.shape[axis+1:])
        if extrap_:
            start = numpy.take(a, [-1], axis=axis)
            ratio = start / numpy.take(a, [-2], axis=axis)
            exp = numpy.arange(1, Npad_ + 1).reshape(to_axis)
            a_ = start * ratio ** exp
        else:
            a_ = numpy.zeros(a.shape[:axis] + (Npad_,) + a.shape[axis+1:])

        return numpy.concatenate((_a, a, a_), axis=axis)


    def _unpad(self, a, axis, out):
        """Undo padding in an array.

        Parameters
        ----------
        a : (..., N, ...) ndarray
            array to be trimmed to length `Nin`
        axis : int
            axis along which to unpad
        out : bool
            trim the output if True, otherwise the input; the two cases have
            their left and right pad sizes reversed
        """

        assert a.shape[axis] == self.N

        Npad = self.N - self.Nin
        if out:
            _Npad, Npad_ = Npad - Npad//2, Npad//2
        else:
            _Npad, Npad_ = Npad//2, Npad - Npad//2

        return numpy.take(a, range(_Npad, self.N - Npad_), axis=axis)


    def check(self, F):
        """Rough sanity checks on the input function.
        """

        assert F.ndim == 1, "checker only supports 1D"

        f = self.xfac * F
        fabs = numpy.abs(f)

        iQ1, iQ3 = numpy.searchsorted(fabs.cumsum(), numpy.array([0.25, 0.75]) * fabs.sum())
        assert 0 != iQ1 != iQ3 != self.Nin, "checker giving up"
        fabs_l = fabs[:iQ1].mean()
        fabs_m = fabs[iQ1:iQ3].mean()
        fabs_r = fabs[iQ3:].mean()

        import warnings
        if fabs_l > fabs_m:
            warnings.warn("left wing seems heavy: {:.2g} vs {:.2g}, "
                    "change tilt and mind convergence".format(fabs_l, fabs_m), RuntimeWarning)
        if fabs_m < fabs_r:
            warnings.warn("right wing seems heavy: {:.2g} vs {:.2g}, "
                    "change tilt and mind convergence".format(fabs_m, fabs_r), RuntimeWarning)

        if fabs[0] > fabs[1]:
            warnings.warn("left tail may blow up: {:.2g} vs {:.2g}, "
                    "change tilt or avoid extrapolation".format(f[0], f[1]), RuntimeWarning)
        if fabs[-2] < fabs[-1]:
            warnings.warn("right tail may blow up: {:.2g} vs {:.2g}, "
                    "change tilt or avoid extrapolation".format(f[-2], f[-1]), RuntimeWarning)

        if f[0]*f[1] <= 0:
            warnings.warn("left tail looks wiggly: {:.2g} vs {:.2g}, "
                    "avoid extrapolation".format(f[0], f[1]), RuntimeWarning)
        if f[-2]*f[-1] <= 0:
            warnings.warn("right tail looks wiggly: {:.2g} vs {:.2g}, "
                    "avoid extrapolation".format(f[-2], f[-1]), RuntimeWarning)
