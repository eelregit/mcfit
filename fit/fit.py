import warnings
from numpy import pi, abs, sign, ceil, exp, log, log2, angle, \
                arange, zeros, concatenate, searchsorted, allclose
from numpy.fft import rfft, hfft
from scipy.optimize import fsolve

class fit(object):
    r"""Fast Integral Transform using the FFTLog [1]_[2]_ algorithm.

    Compute integral transforms of the form

    .. math:: G(y) = \int_0^\infty F(x) K(xy) \,\frac{\mathrm{d}x}x
    .. math:: g(y) = \int_0^\infty f(x) (xy)^q K(xy) \,\frac{\mathrm{d}x}x

    using FFTLog, where :math:`K` is a kernel function.

    Parameters
    ----------
    x : float, array_like
        logarithmically uniform abscissae of F
    F : float, array_like
        input function tabulated at x, internally padded symmetrically to length N
        with power-law extrapolations

    UK : callable
        Mellin transform of the kernel
        .. math:: U_K(z) \equiv \int_0^\infty t^z K(t) \, \mathrm{d}t
    q : float
        power-law tilt, can be used to balance f at large and small x.
        avoid the singularities in UK

    N : int, optional
        length of 1D FFT, defaults to the smallest 2**(integer-1) >= len(x)
        so the padding is at least a half
    lowring : bool, optional
        if True and N is even, set the y range according to the low-ringing condition,
        otherwise :math:`x_\textrm{min}*y_\textrm{max}=x_\textrm{max}*y_\textrm{min}=1`

    prefac : float, array_like
        prefactor array to multiply before the transform,
        serves to convert a particular transform to the general form,
        e.g. prefac=x**2 for Hankel transform
    postfac : float, array_like
        array to multiply after the transform is done,
        serves to convert a particular transform to the general form,

    Returns
    -------
    y : float, array_like
        logarithmically uniform abscissae of G
    G : float, array_like
        output function tabulated at y, internal paddings are discarded before output

    References
    ----------
    .. [1] J. D. Talman. Numerical Fourier and Bessel Transforms in Logarithmic Variables.
           Journal of Computational Physics, 29:35-48, October 1978.
    .. [2] A. J. S. Hamilton. Uncorrelated modes of the non-linear power spectrum.
           MNRAS, 312:257-284, February 2000.
    """

    def __init__(self, x, UK, q, N=None, lowring=True, prefac=1, postfac=1):
        self.UK = UK
        self.q = q
        self._set_range(x, N, lowring)
        self.prefac = prefac
        self.postfac = postfac
        self._calc_coeff()


    def _set_range(self, x, N, lowring):
        self.Ndata = len(x)
        assert self.Ndata > 1
        self.Delta = log(x[-1] / x[0]) / (self.Ndata - 1)
        self.ratio = exp(self.Delta)
        assert allclose(x[1:] / x[:-1], self.ratio, rtol=1e-3), "abscissae not log-uniform"

        if N is None:
            folds = int(ceil(log2(self.Ndata))) + 1
            self.N = 2**folds
        else:
            self.N = N
        assert self.N >= self.Ndata
        self.Npad = self.N - self.Ndata

        self.lowring = lowring
        self.lnxy = 0  # = lnxmin + lnymax = lnxmax + lnymin
        if lowring and self.N % 2 == 0:
            self.lnxy = self.Delta / pi * angle(self.UK(self.q + 1j * pi / self.Delta))
        self.x = x
        self.y = exp(self.lnxy - self.Delta) / x[::-1]


    def _calc_coeff(self):
        m = arange(0, self.N//2 + 1)
        self.u = self.UK(self.q + 2j * pi / self.N / self.Delta * m)
        self.u *= exp(-2j * pi * self.lnxy /self.N /self.Delta * m)

        # following is unnecessary, as hfft ignore its imag anyway
        # if not self.lowring and self.N % 2 == 0:
        #     self.u[self.N//2] = self.u[self.N//2].real


    def __call__(self, F, extrap=True):
        f = self.prefac * self.x**(-self.q) * F

        # pad with power-law extrapolation, or zeros
        if extrap:
            fpad_l = f[0] * (f[1] / f[0]) ** arange(-(self.Npad//2), 0)
            fpad_r = f[-1] * (f[-1] / f[-2]) ** arange(1, self.Npad - self.Npad//2 + 1)
        else:
            fpad_l = zeros(self.Npad//2)
            fpad_r = zeros(self.Npad - self.Npad//2)
        f = concatenate((fpad_l, f, fpad_r))

        # convolution
        f = rfft(f)  # f(x_n) -> f_m
        g = f * self.u  # f_m -> g_m
        g = hfft(g, self.N) / self.N  # g_m -> g(y_n)

        # back to the trusted range
        g = g[self.Npad - self.Npad//2 : self.N - self.Npad//2]

        G = self.postfac * self.y**(-self.q) * g

        return self.y, G


    def check(self, F):
        """sanity checks
        """
        f = self.prefac * self.x**(-self.q) * F
        fabs = abs(f)

        # reckon q range
        iQ1, iQ2, iQ3 = searchsorted(fabs.cumsum(), arange(1,4) / 4 * fabs.sum())
        assert 0 != iQ1 != iQ2 != iQ3 != self.Ndata, "checker giving up"
        fabs0 = fabs[:iQ1].mean()
        fabs1 = fabs[iQ1:iQ2].mean()
        fabs2 = fabs[iQ2:iQ3].mean()
        fabs3 = fabs[iQ3:].mean()
        fabs_imba_l = fabs1 * (iQ2 - iQ1) / fabs0 / iQ1
        fabs_imba_r = fabs3 * (self.Ndata - iQ3) / fabs2 / (iQ3 - iQ2)
        def eqn_q_l(x):
            return (exp(iQ2*x) - exp(iQ1*x)) / (exp(iQ1*x) - 1) - fabs_imba_l
        x0_l = sign(fabs1 - fabs0) / iQ2
        niceq_l = fsolve(eqn_q_l, x0_l)[0] / self.Delta + self.q
        def eqn_q_r(x):
            return (exp(self.Ndata*x) - exp(iQ3*x)) / (exp(iQ3*x) - exp(iQ2*x)) - fabs_imba_r
        x0_r = sign(fabs3 - fabs2) / self.Ndata
        niceq_r = fsolve(eqn_q_r, x0_r)[0] / self.Delta + self.q
        if fabs0 > fabs1:
            print("left wing seems heavy: {:.2g} vs {:.2g}".format(fabs0, fabs1))
        if fabs2 < fabs3:
            print("right wing seems heavy: {:.2g} vs {:.2g}".format(fabs2, fabs3))
        print("tilt q ∊ ({:.2f}, {:.2f}) may balance well, "
                "current choice q = {}".format(niceq_r, niceq_l, self.q))
        if niceq_l < niceq_r:
            warnings.warn("checker questioning convergence")

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
