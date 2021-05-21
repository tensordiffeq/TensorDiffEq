import numpy as np


def u_analytical(flux_type, M, x, t):
    # indx_zero = x == 0.0
    # x = x[~indx_zero]
    # t = t[~indx_zero]
    if flux_type == 'concave':
        u = concave(M, x, t)
    elif flux_type == 'non-convex':
        u = nonconvex(M, x, t)
    else:  # convex
        u = np.zeros(x.shape)
        u = np.where(x <= t, 1.0, 0.0)

    return u


def concave(M, x, t):
    u = np.zeros(x.shape)
    for i in range(x.shape[0]):
        if t[i] == 0 or x[i] == 0:
            u[i] = 0.0
        else:
            x_t = x[i] / t[i]
            if x_t > M:
                u[i] = 0
            elif x_t > 1 / M:
                u[i] = (np.sqrt(M / x_t) - 1.0) / (M - 1)
            else:
                u[i] = 1.0
    return u


def nonconvex(M, x, t):
    # soluiton from http://www.clawpack.org/riemann_book/html/Nonconvex_scalar.html
    a = 1. / M
    q_left = 1.
    q_right = 0.

    ind_x = t == 0.
    xi = np.zeros(x.shape)
    xi[~ind_x] = x[~ind_x] / t[~ind_x]

    f_buckley_leverett = lambda q: q ** 2 / (q ** 2 + a * (1 - q) ** 2)

    qxi = nonconvex_solutions(f_buckley_leverett, q_left, q_right, xi)

    return qxi


def nonconvex_solutions(f, q_left, q_right, xi):
    """
    Compute the Riemann solution to a scalar conservation law.

    Compute the similarity solution Q(x/t) and also the
    (possibly multi-valued) solution determined by tracing
    characteristics.

    Input:
      f = flux function (possibly nonconvex)
      q_left, q_right = Riemann data
      xi_left, xi_right = optional left and right limits for xi = x/t
               in similarity solution.
               If not specified, chosen based on the characteristic speeds.

    Returns:
      xi = array of values between xi_left and xi_right
      q  = array of corresponding q(xi) values (xi = x/t)
      q_char = array of values of q between q_left and q_right
      xi_char = xi value for each q_char for use in plotting the
              (possibly multi-valued) solution where each q value
              propagates at speed f'(q).
    """

    from numpy import linspace, diff, hstack

    qtilde = osher_solution(f, q_left, q_right)

    q = qtilde(xi)

    return q


def osher_solution(f, q_left, q_right, n=1000):
    """
    Compute the Riemann solution to a scalar conservation law.

    Compute the similarity solution Q(x/t) and also the
    (possibly multi-valued) solution determined by tracing
    characteristics.

    Input:
      f = flux function (possibly nonconvex)
      q_left, q_right = Riemann data

    Returns:
      qtilde = function of xi = x/t giving the Riemann solution
    """

    from numpy import linspace, empty, argmin, argmax

    q_min = min(q_left, q_right)
    q_max = max(q_left, q_right)
    qv = linspace(q_min, q_max, n)

    # define the function qtilde as in (16.7)
    if q_left <= q_right:
        def qtilde(xi):
            Q = empty(xi.shape, dtype=float)
            for j, xij in enumerate(xi):
                i = argmin(f(qv) - xij * qv)
                Q[j] = qv[i]
            return Q
    else:
        def qtilde(xi):
            Q = empty(xi.shape, dtype=float)
            for j, xij in enumerate(xi):
                i = argmax(f(qv) - xij * qv)
                Q[j] = qv[i]
            return Q

    return qtilde
