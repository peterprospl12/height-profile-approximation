def lagrange_interpolation_cython(list x, list y, list x_new):
    cdef int n = len(x)
    cdef int m = len(x_new)
    cdef list y_new = [0.0] * m
    cdef int i, j, k
    cdef double p

    for i in range(m):
        for j in range(n):
            p = 1.0
            for k in range(n):
                if k != j and x[j] - x[k] != 0:
                    p *= (x_new[i] - x[k]) / (x[j] - x[k])
            y_new[i] += y[j] * p

    return y_new