import math
import cmath
import warnings

import numpy
try:
    import jax
    jax.config.update("jax_enable_x64", True)
except ModuleNotFoundError as e:
    JAXNotFoundError = e


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
        log-spaced input argument
    MK : callable
        Mellin transform of the kernel
        .. math:: U_K(z) \equiv \int_0^\infty t^{z-1} K(t) dt
    q : float
        power-law tilt, can be used to balance :math:`f` at large and small
        :math:`x`. Avoid the singularities in `MK`
    N : int or complex, optional
        size of convolution, if complex then replaced by the smallest power of
        2 that is at least `N.imag` times the size of `x`; the input function
        is padded symmetrically to this size before convolution (see the
        `extrap` argument for available options); `N=len(x)` turns off the
        padding
    lowring : bool, optional
        if True and `N` is even, set `y` according to the low-ringing
        condition, otherwise see `xy`
    xy : float, optional
        reciprocal product :math:`x_{min} y_{max} = x_{max} y_{min}` to be used
        when `lowring` is False or `N` is odd.
        `xy = x[0] * y_max = x[1] * y[-1] = ... = x[i] * y[-i] = ... = x[-1] * y[1] = x_max * y[0]`.
        Note that :math:`x_{max}` is not included in `x` but bigger than
        `x.max()` by one log interval due to the discretization of the periodic
        approximant, and likewise for :math:`y_{max}`
    backend : str in {'numpy', 'jax'}, optional
        Which backend to use.

    Attributes
    ----------
    Nin : int
        input size, and that of the output if not `keeppads`
    N : int
        convolution size, and that of the output if `keeppads`
    x : (Nin,) ndarray
        input argument
    y : (Nin,) ndarray
        output argument
    _x_ : (N,) ndarray
        padded `x`
    _y_ : (N,) ndarray
        padded `y`
    xy : float
        reciprocal product
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

    Examples
    --------
    >>> x = numpy.logspace(-3, 3, num=60, endpoint=False)
    >>> A = 1 / (1 + x*x)**1.5
    >>> H = mcfit.mcfit(x, mcfit.kernels.Mellin_BesselJ(0), q=1, lowring=True)
    >>> y, B = H(x**2 * A, extrap=True)
    >>> numpy.allclose(B, numpy.exp(-y))

    More conveniently, use the Hankel transform subclass
    >>> y, B = mcfit.transforms.Hankel(x, lowring=True)(A, extrap=True)

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

    def __init__(self, x, MK, q, N=2j, lowring=False, xy=1, backend='numpy'):
        if backend == 'numpy':
            self.np = numpy
            #self.jit = lambda fun: fun  # TODO maybe use Numba?
        elif backend == 'jax':
            try:
                self.np = jax.numpy
                #self.jit = jax.jit  # TODO maybe leave it to the user? jax.jit for CPU too
            except NameError:
                raise JAXNotFoundError
        else:
            raise ValueError(f"backend {backend} not supported")

        #self.__call__ = self.jit(self.__call__)
        #self.matrix = self.jit(self.matrix)

        self.x = self.np.asarray(x)
        self.Nin = len(x)
        self.MK = MK
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
            raise ValueError(f"input size {self.Nin} must not be smaller than 2")
        Delta = math.log(self.x[-1] / self.x[0]) / (self.Nin - 1)
        x_head = self.x[:8]
        if not self.np.allclose(self.np.log(x_head[1:] / x_head[:-1]), Delta,
                                rtol=1e-3):
            warnings.warn("input must be log-spaced")

        if isinstance(self.N, complex):
            folds = math.ceil(math.log2(self.Nin * self.N.imag))
            self.N = 2**folds
        if self.N < self.Nin:
            raise ValueError(f"convolution size {self.N} must not be smaller than "
                             f"the input size {self.Nin}")

        if self.lowring and self.N % 2 == 0:
            lnxy = Delta / math.pi * cmath.phase(self.MK(self.q + 1j * math.pi / Delta))
            self.xy = math.exp(lnxy)
        else:
            lnxy = math.log(self.xy)
        self.y = math.exp(lnxy - Delta) / self.x[::-1]

        self._x_ = self._pad(self.x, 0, True, False)
        self._y_ = self._pad(self.y, 0, True, True)

        m = numpy.arange(0, self.N//2 + 1)
        self._u = self.MK(self.q + 2j * math.pi / self.N / Delta * m)
        self._u *= numpy.exp(-2j * math.pi * lnxy / self.N / Delta * m)
        self._u = self.np.asarray(self._u, dtype=(self.x[0] + 0j).dtype)

        # following is unnecessary because hfft ignores the imag at Nyquist anyway
        #if not self.lowring and self.N % 2 == 0:
        #    self._u[self.N//2] = self._u[self.N//2].real

    def __call__(self, F, axis=-1, extrap=False, keeppads=False, convonly=False):
        """Evaluate the integral.

        Parameters
        ----------
        F : (..., Nin, ...) or (..., N, ...) array_like
            input function; to be padded according to `extrap` in size from
            `Nin` to `N`, but not if already of size `N`
        axis : int, optional
            axis along which to integrate
        extrap : {bool, 'const'} or 2-tuple, optional
            Method to extrapolate `F`.
            For a 2-tuple, the two elements are for the left and right pads,
            whereas a single value applies to both ends.
            Options are:
            * True: power-law extrapolation using the end segment
            * False: zero padding
            * 'const': constant padding with the end point value
        keeppads : bool, optional
            whether to keep the padding in the output
        convonly : bool, optional
            whether to skip the scaling by `_xfac_` and `_yfac_`, useful for
            evaluating integral with multiple kernels

        Returns
        -------
        y : (Nin,) or (N,) ndarray
            log-spaced output argument
        G : (..., Nin, ...) or (..., N, ...) ndarray
            output function

        """
        F = self.np.asarray(F)

        to_axis = [1] * F.ndim
        to_axis[axis] = -1

        f = self._pad(F, axis, extrap, False)
        if not convonly:
            f = self._xfac_.reshape(to_axis) * f

        # convolution
        f = self.np.fft.rfft(f, axis=axis)  # f(x_n) -> f_m
        g = f * self._u.reshape(to_axis)  # f_m -> g_m
        g = self.np.fft.hfft(g, n=self.N, axis=axis) / self.N  # g_m -> g(y_n)

        if not keeppads:
            G = self._unpad(g, axis, True)
            if not convonly:
                G = self.yfac.reshape(to_axis) * G
            return self.y, G
        else:
            _G_ = g
            if not convonly:
                _G_ = self._yfac_.reshape(to_axis) * _G_
            return self._y_, _G_

    def inv(self):
        """Invert the transform.

        After calling this method, calling the instance will do the inverse
        transform. Calling this twice return the instance to the original
        transform.

        """
        self.x, self.y = self.y, self.x
        self._x_, self._y_ = self._y_, self._x_
        self.xfac, self.yfac = 1 / self.yfac, 1 / self.xfac
        self._xfac_, self._yfac_ = 1 / self._yfac_, 1 / self._xfac_
        self._u = 1 / self._u.conj()

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
        `M`, `a`, `b`, and `C` are padded by default.

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
        v = self.np.fft.hfft(self._u, n=self.N) / self.N
        idx = sum(self.np.ogrid[0:self.N, -self.N:0])
        C = v[idx]  # follow scipy.linalg.{circulant,toeplitz,hankel}

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
        """Add padding to an array.

        Parameters
        ----------
        a : (..., Nin, ...) or (..., N, ...) ndarray
            array to be padded, but not if already of size `N`
        axis : int
            axis along which to pad
        extrap : {bool, 'const'} or 2-tuple
            Method to extrapolate `a`.
            For a 2-tuple, the two elements are for the left and right pads,
            whereas a single value applies to both ends.
            Options are:
            * True: power-law extrapolation using the end segment
            * False: zero padding
            * 'const': constant padding with the end point value
        out : bool
            pad the output if True, otherwise the input; the two cases have
            their left and right pad sizes reversed

        """
        if a.shape[axis] == self.N:
            return a
        elif a.shape[axis] != self.Nin:
            raise ValueError("array size must be that of the input or the convolution")

        axis %= a.ndim  # to fix the indexing below with axis+1

        to_axis = [1] * a.ndim
        to_axis[axis] = -1

        Npad = self.N - self.Nin
        if out:
            _Npad, Npad_ = Npad - Npad//2, Npad//2
        else:
            _Npad, Npad_ = Npad//2, Npad - Npad//2

        try:
            _extrap, extrap_ = extrap
        except (TypeError, ValueError):
            _extrap = extrap_ = extrap

        if isinstance(_extrap, bool):
            if _extrap:
                end = self.np.take(a, self.np.array([0]), axis=axis)
                ratio = self.np.take(a, self.np.array([1]), axis=axis) / end
                exp = self.np.arange(-_Npad, 0).reshape(to_axis)
                _a = end * ratio ** exp
            else:
                _a = self.np.zeros(a.shape[:axis] + (_Npad,) + a.shape[axis+1:])
        elif _extrap == 'const':
            end = self.np.take(a, self.np.array([0]), axis=axis)
            _a = self.np.repeat(end, _Npad, axis=axis)
        else:
            raise ValueError(f"left extrap {_extrap} not supported")
        if isinstance(extrap_, bool):
            if extrap_:
                end = self.np.take(a, self.np.array([-1]), axis=axis)
                ratio = end / self.np.take(a, self.np.array([-2]), axis=axis)
                exp = self.np.arange(1, Npad_ + 1).reshape(to_axis)
                a_ = end * ratio ** exp
            else:
                a_ = self.np.zeros(a.shape[:axis] + (Npad_,) + a.shape[axis+1:])
        elif extrap_ == 'const':
            end = self.np.take(a, self.np.array([-1]), axis=axis)
            a_ = self.np.repeat(end, Npad_, axis=axis)
        else:
            raise ValueError(f"right extrap {extrap_} not supported")

        return self.np.concatenate((_a, a, a_), axis=axis)

    def _unpad(self, a, axis, out):
        """Undo padding in an array.

        Parameters
        ----------
        a : (..., N, ...) or (..., Nin, ...) ndarray
            array to be unpadded, but not if already of size `Nin`
        axis : int
            axis along which to unpad
        out : bool
            unpad the output if True, otherwise the input; the two cases have
            their left and right pad sizes reversed

        """
        if a.shape[axis] == self.Nin:
            return a
        elif a.shape[axis] != self.N:
            raise ValueError("array size must be that of the input or the convolution")

        Npad = self.N - self.Nin
        if out:
            _Npad, Npad_ = Npad - Npad//2, Npad//2
        else:
            _Npad, Npad_ = Npad//2, Npad - Npad//2

        return self.np.take(a, self.np.arange(_Npad, self.N - Npad_), axis=axis)
