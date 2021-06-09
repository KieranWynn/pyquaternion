import functools
__author__ = 'Alex Pyattaev'
import os
import numba.core.errors
try:

    if 'NO_NUMBA' in os.environ:
        raise ImportError("Numba is disabled")

    import numba
    import numba.experimental
    numba_available = True
    jit_hardcore = functools.partial(numba.jit, nopython=True, nogil=True, cache=True)
    jit = functools.partial(numba.jit, forceobj=True, nopython=False, cache=True)
    jitclass = numba.experimental.jitclass
    int64 = numba.int64
    int16 = numba.int16
    double = numba.double
    complex128 = numba.complex128
    TypingError = numba.core.errors.TypingError
except ImportError:
    numba = None
    numba_available = False
    TypingError = TypeError
    int64 = int
    int16 = int
    double = float
    complex128 = complex

    #define stub functions for Numba placeholders
    def jit(f, *args, **kwargs):
        return f

    def jitclass(c, *args, **kwargs):
        def x(cls):
            return cls
        return x

    def jit_hardcore(f, *args, **kwargs):
        return f


