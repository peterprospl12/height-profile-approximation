import numpy as np
cimport numpy as np

def lagrange_interpolation_cython(np.ndarray[np.float64_t, ndim=1] x,
                                   np.ndarray[np.float64_t, ndim=1] y,
                                   np.ndarray[np.float64_t, ndim=1] x_new):
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t m = x_new.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] y_new = np.zeros(m, dtype=np.float64)
    cdef Py_ssize_t i, j, k
    cdef double p

    for i in range(m):
        for j in range(n):
            p = 1.0
            for k in range(n):
                if k != j and (x[j] - x[k]) != 0:
                    p *= (x_new[i] - x[k]) / (x[j] - x[k])
            y_new[i] += y[j] * p

    return y_new
