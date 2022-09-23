import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import numpy as np
import platform


def fem1d_bvp_quadratic(n, a, c, f, x):

    # *****************************************************************************80
    #
    # fem1d_bvp_quadratic() solves a two point boundary value problem.
    #
    #  Discussion:
    #
    #    The program uses the finite element method, with piecewise quadratic basis
    #    functions to solve a boundary value problem in one dimension.
    #
    #    The problem is defined on the region 0 <= x <= 1.
    #
    #    The following differential equation is imposed between 0 and 1:
    #
    #      - d/dx a(x) du/dx + c(x) * u(x) = f(x)
    #
    #    where a(x), c(x), and f(x) are given functions.
    #
    #    At the boundaries, the following conditions are applied:
    #
    #      u(0.0) = 0.0
    #      u(1.0) = 0.0
    #
    #    A set of N equally spaced nodes is defined on this
    #    interval, with 0 = X(1) < X(2) < ... < X(N) = 1.0.
    #
    #    At each node I, we associate a piecewise quadratic basis function V(I,X),
    #    which is 0 at all nodes except node I.
    #
    #    We now assume that the solution U(X) can be written as a quadratic
    #    sum of these basis functions:
    #
    #      U(X) = sum ( 1 <= J <= N ) U(J) * V(J,X)
    #
    #    where U(X) on the left is the function of X, but on the right,
    #    is meant to indicate the coefficients of the basis functions.
    #
    #    To determine the coefficient U(J), we multiply the original
    #    differential equation by the basis function V(J,X), and use
    #    integration by parts, to arrive at the I-th finite element equation:
    #
    #        Integral A(X) * U'(X) * V'(I,X) + C(X) * U(X) * V(I,X) dx
    #      = Integral F(X) * V(I,X) dx
    #
    #    By writing this equation for basis functions I = 2 through N - 1,
    #    and using the boundary conditions, we have N linear equations
    #    for the N unknown coefficients U(1) through U(N), which can
    #    be easily solved.
    #

    #  Input:
    #
    #    integer N, the number of nodes.
    #
    #    function A ( X ), evaluates a(x);
    #
    #    function C ( X ), evaluates c(x);
    #
    #    function F ( X ), evaluates f(x);
    #
    #    real X(N), the mesh points.
    #
    #  Output:
    #
    #    real U(N), the finite element coefficients, which are also
    #    the value of the computed solution at the mesh points.
    #

    #
    #  Define a 3 point Gauss-Legendre quadrature rule on [-1,+1].
    #
    quad_num = 3
    abscissa = np.array([
        -0.774596669241483377035853079956,
        0.000000000000000000000000000000,
        0.774596669241483377035853079956])
    weight = np.array([
        0.555555555555555555555555555556,
        0.888888888888888888888888888889,
        0.555555555555555555555555555556])
#
#  Make room for the matrix A and right hand side b.
#
    A = np.zeros([n, n])
    b = np.zeros(n)
#
#  Integrate over element E.
#
    e_num = (n - 1) // 2

    for e in range(0, e_num):

        l = 2 * e
        xl = x[l]

        m = 2 * e + 1
        xm = x[m]

        r = 2 * e + 2
        xr = x[r]

        for q in range(0, quad_num):

            xq = ((1.0 - abscissa[q]) * xl
                  + (1.0 + abscissa[q]) * xr) \
                / 2.0

            wq = weight[q] * (xr - xl) / 2.0

            axq = a(xq)
            cxq = c(xq)
            fxq = f(xq)

            vl = ((xq - xm) / (xl - xm)) \
                * ((xq - xr) / (xl - xr))

            vm = ((xq - xl) / (xm - xl)) \
                * ((xq - xr) / (xm - xr))

            vr = ((xq - xl) / (xr - xl)) \
                * ((xq - xm) / (xr - xm))

            vlp = (1.0 / (xl - xm)) \
                * ((xq - xr) / (xl - xr)) \
                + ((xq - xm) / (xl - xm)) \
                * (1.0 / (xl - xr))

            vmp = (1.0 / (xm - xl)) \
                * ((xq - xr) / (xm - xr)) \
                + ((xq - xl) / (xm - xl)) \
                * (1.0 / (xm - xr))

            vrp = (1.0 / (xr - xl)) \
                * ((xq - xm) / (xr - xm)) \
                + ((xq - xl) / (xr - xl)) \
                * (1.0 / (xr - xm))

            A[l, l] = A[l, l] + wq * (vlp * axq * vlp + vl * cxq * vl)
            A[l, m] = A[l, m] + wq * (vlp * axq * vmp + vl * cxq * vm)
            A[l, r] = A[l, r] + wq * (vlp * axq * vrp + vl * cxq * vr)
            b[l] = b[l] + wq * (vl * fxq)

            A[m, l] = A[m, l] + wq * (vmp * axq * vlp + vm * cxq * vl)
            A[m, m] = A[m, m] + wq * (vmp * axq * vmp + vm * cxq * vm)
            A[m, r] = A[m, r] + wq * (vmp * axq * vrp + vm * cxq * vr)
            b[m] = b[m] + wq * (vm * fxq)

            A[r, l] = A[r, l] + wq * (vrp * axq * vlp + vr * cxq * vl)
            A[r, m] = A[r, m] + wq * (vrp * axq * vmp + vr * cxq * vm)
            A[r, r] = A[r, r] + wq * (vrp * axq * vrp + vr * cxq * vr)
            b[r] = b[r] + wq * (vr * fxq)
#
#  Equation 0 is the left boundary condition, U(0) = 0.0;
#
    for j in range(0, n):
        A[0, j] = 0.0
    A[0, 0] = 1.0
    b[0] = 0.0
#
#  We can keep the matrix symmetric by using the boundary condition
#  to eliminate U(0) from all equations but #0.
#
    for i in range(1, n):
        b[i] = b[i] - A[i, 0] * b[0]
        A[i, 0] = 0.0
#
#  Equation N-1 is the right boundary condition, U(N-1) = 0.0;
#
    for j in range(0, n):
        A[n-1, j] = 0.0
    A[n-1, n-1] = 1.0
    b[n-1] = 0.0
#
#  We can keep the matrix symmetric by using the boundary condition
#  to eliminate U(N-1) from all equations but #N-1.
#
    for i in range(0, n - 1):
        b[i] = b[i] - A[i, n-1] * b[n-1]
        A[i, n-1] = 0.0
#
#  Solve the linear system for the finite element coefficients U.
#
    u = la.solve(A, b)

    return u


def fem1d_bvp_quadratic_test00():

    n = 10

#
#  Geometry definitions.
#
    x_lo = 0.0
    x_hi = 1.0
    x = np.linspace(x_lo, x_hi, n)

    u = fem1d_bvp_quadratic(n, a00, c00, f00, x)

    g = np.zeros(n)
    for i in range(0, n):
        g[i] = exact00(x[i])
#
#  Print a table.
#
    print('')
    print('     I    X         U         Uexact    Error')
    print('')

    for i in range(0, n):
        print('  %4d  %8f  %8f  %8f  %8e'
              % (i, x[i], u[i], g[i], abs(u[i] - g[i])))
#
#  Compute error norms.
#
    e1 = l1_error(n, x, u, exact00)
    e2 = l2_error_quadratic(n, x, u, exact00)
    h1s = h1s_error_quadratic(n, x, u, exactp00)
    mx = max_error_quadratic(n, x, u, exact00)

    print('')
    print('  l1 norm of error  = %g' % (e1))
    print('  L2 norm of error  = %g' % (e2))
    print('  Seminorm of error = %g' % (h1s))
    print('  Max norm of error = %g' % (mx))
#
#  Plot the computed solution.
#
    fig = plt.figure()
    plt.plot(x, u, 'bo-')
    plt.xlabel('<---X--->')
    plt.ylabel('<---U(X)--->')
    plt.title('fem1d_bvp_quadratic_test00 Solution')
    plt.savefig('fem1d_bvp_quadratic_test00.png')
    plt.show()
    plt.close()
#
#  Terminate.
#
    print('')
    print('fem1d_bvp_quadratic_test00')
    print('  Normal end of execution.')
    return


def a00(x):

    # *****************************************************************************80
    #
    # a00() evaluates the A coefficient.

    #
    value = 1.0

    return value


def c00(x):

    # *****************************************************************************80
    #
    # c00() evaluates the C coefficient.

    value = 1.0
    return value


def exact00(x):

    # *****************************************************************************80
    #
    # exact00() evaluates the exact solution.

    value = (1 - x) * (np.arctan(a * (x - xb)) + np.arctan(a*xb))

    return value


def exactp00(x):

    # *****************************************************************************80
    #
    # exactp00() evaluates the derivative of the exact solution.

    B = x-xb
    value = 2*(a+a**3*B*(B-x+1))/(a**2*B**2+1)**2
    return value


def f00(x):

    # *****************************************************************************80
    #
    # f00() evaluates the right hand side.

    value = x
    return value


def l1_error(n, x, u, exact):

    # *****************************************************************************80
    #
    # l1_error() estimates the l1 error norm of a finite element solution.
    #
    #  Discussion:
    #
    #    We assume the finite element method has been used, over an interval [A,B]
    #    involving N nodes.
    #
    #    The coefficients U(1:N) have been computed, and a formula for the
    #    exact solution is known.
    #
    #    This function estimates the little l1 norm of the error:
    #      L1_NORM = sum ( 1 <= I <= N ) abs ( U(i) - EXACT(X(i)) )
    #
    #  Input:
    #
    #    integer N, the number of nodes.
    #
    #    real X(N), the mesh points.
    #
    #    real U(N), the finite element coefficients.
    #
    #    function EQ = EXACT ( X ), returns the value of the exact
    #    solution at the point X.
    #
    #  Output:
    #
    #    real E1, the little l1 norm of the error.
    #
    e1 = 0.0
    for i in range(0, n):
        e1 = e1 + abs(u[i] - exact(x[i]))

    e1 = e1 / n

    return e1


def l2_error_quadratic(n, x, u, exact):

    # *****************************************************************************80
    #
    # l2_error_quadratic() estimates the L2 error norm of a finite element solution.
    #
    #  Discussion:
    #
    #    We assume the finite element method has been used, over an interval [A,B]
    #    involving N nodes, with piecewise quadratic elements used for the basis.
    #    The coefficients U(1:N) have been computed, and a formula for the
    #    exact solution is known.
    #
    #    This function estimates the L2 norm of the error:
    #
    #      L2_NORM = Integral ( A <= X <= B ) ( U(X) - EXACT(X) )^2 dX
    #

    #
    #  Input:
    #
    #    integer N, the number of nodes.
    #
    #    real X(N), the mesh points.
    #
    #    real U(N), the finite element coefficients.
    #
    #    function EQ = EXACT ( X ), returns the value of the exact
    #    solution at the point X.
    #
    #  Output:
    #
    #    real E2, the estimated L2 norm of the error.
    #
    import numpy as np

    e2 = 0.0
#
#  Define a 2 point Gauss-Legendre quadrature rule on [-1,+1].
#
    quad_num = 3
    abscissa = np.array([
        -0.774596669241483377035853079956,
        0.000000000000000000000000000000,
        0.774596669241483377035853079956])
    weight = np.array([
        0.555555555555555555555555555556,
        0.888888888888888888888888888889,
        0.555555555555555555555555555556])
#
#  Integrate over each interval.
#
    e_num = (n - 1) // 2

    for e in range(0, e_num):

        l = 2 * e
        xl = x[l]
        ul = u[l]

        m = 2 * e + 1
        xm = x[m]
        um = u[m]

        r = 2 * e + 2
        xr = x[r]
        ur = u[r]

        for q in range(0, quad_num):

            xq = ((1.0 - abscissa[q]) * xl
                  + (1.0 + abscissa[q]) * xr) \
                / 2.0

            wq = weight[q] * (xr - xl) / 2.0

            vl = ((xq - xm) / (xl - xm)) \
                * ((xq - xr) / (xl - xr))

            vm = ((xq - xl) / (xm - xl)) \
                * ((xq - xr) / (xm - xr))

            vr = ((xq - xl) / (xr - xl)) \
                * ((xq - xm) / (xr - xm))

            uq = u[l] * vl + u[m] * vm + u[r] * vr

            eq = exact(xq)

            e2 = e2 + wq * (uq - eq) ** 2

    e2 = np.sqrt(e2)

    return e2


def l2_error_quadratic_test():

    # *****************************************************************************80
    #
    # l2_error_quadratic_test() tests l2_error_quadratic().
    #

    x_n = 3

    for test in range(0, 6):
        x_lo = 0.0
        x_hi = np.pi
        x = np.linspace(x_lo, x_hi, x_n)
#
#  U is an approximation to sin(x).
#
        u = np.zeros(x_n)
        for i in range(0, x_n):
            u[i] = np.sin(x[i])

        e1 = l2_error_quadratic(x_n, x, u, np.sin)

        print('  %2d  %g' % (x_n, e1))

        x_n = 2 * (x_n - 1) + 1
#
#  Terminate.
#
    print('')
    print('l2_error_quadratic_test:')
    print('  Normal end of execution.')
    return


def max_error_quadratic(n, x, u, exact):

    # *****************************************************************************80
    #
    # max_error_quadratic() estimates the max error norm of a finite element solution.
    #
    #  Discussion:
    #
    #    We assume the finite element method has been used, over an interval [A,B]
    #    involving N nodes, with piecewise quadratic elements used for the basis.
    #    The coefficients U(1:N) have been computed, and a formula for the
    #    exact solution is known.
    #
    #    This function estimates the max norm of the error:
    #
    #      MAX_NORM = Integral ( A <= X <= B ) max ( abs ( U(X) - EXACT(X) ) ) dX
    #

    #
    #  Input:
    #
    #    integer N, the number of nodes.
    #
    #    real X(N), the mesh points.
    #
    #    real U(N), the finite element coefficients.
    #
    #    function EQ = EXACT ( X ), returns the value of the exact
    #    solution at the point X.
    #
    #  Output:
    #
    #    real VALUE, the estimated max norm of the error.
    #
    import numpy as np

    quad_num = 8
    value = 0.0
#
#  Examine QUAD_NUM points in each element, including left node but not right.
#
    e_num = (n - 1) // 2

    for e in range(0, e_num):

        l = 2 * e
        xl = x[l]
        ul = u[l]

        m = 2 * e + 1
        xm = x[m]
        um = u[m]

        r = 2 * e + 2
        xr = x[r]
        ur = u[r]

        for q in range(0, quad_num):

            xq = (float(quad_num - q) * xl
                  + float(q) * xr) \
                / float(quad_num)

            vl = ((xq - xm) / (xl - xm)) \
                * ((xq - xr) / (xl - xr))

            vm = ((xq - xl) / (xm - xl)) \
                * ((xq - xr) / (xm - xr))

            vr = ((xq - xl) / (xr - xl)) \
                * ((xq - xm) / (xr - xm))

            uq = u[l] * vl + u[m] * vm + u[r] * vr

            eq = exact(xq)

            value = max(value, abs(uq - eq))
#
#  For completeness, check last node.
#
    xq = x[n-1]
    uq = u[n-1]
    eq = exact(xq)

    value = max(value, abs(uq - eq))
#
#  Integral approximation requires multiplication by interval length.
#
    value = value * (x[n-1] - x[0])

    return value


def h1s_error_quadratic(n, x, u, exact_ux):

    # *****************************************************************************80
    #
    # h1s_error_quadratic(): seminorm error of a finite element solution.
    #
    #  Discussion:
    #
    #    We assume the finite element method has been used, over an interval [A,B]
    #    involving N nodes, with piecewise quadratic elements used for the basis.
    #    The finite element solution U(x) has been computed, and a formula for the
    #    exact derivative V'(x) is known.
    #
    #    This function estimates the H1 seminorm of the error:
    #
    #      H1S = sqrt ( integral ( A <= x <= B ) ( U'(x) - V'(x) )^2 dx
    #

    #
    #  Input:
    #
    #    integer N, the number of nodes.
    #
    #    real X(N), the mesh points.
    #
    #    real U(N), the finite element coefficients.
    #
    #    function EQ = EXACT_UX ( X ), returns the value of the exact
    #    derivative at the point X.
    #
    #  Output:
    #
    #    real H1S, the estimated seminorm of the error.
    #
    import numpy as np

    h1s = 0.0
#
#  Define a 2 point Gauss-Legendre quadrature rule on [-1,+1].
#
    quad_num = 3
    abscissa = np.array([
        -0.774596669241483377035853079956,
        0.000000000000000000000000000000,
        0.774596669241483377035853079956])
    weight = np.array([
        0.555555555555555555555555555556,
        0.888888888888888888888888888889,
        0.555555555555555555555555555556])
#
#  Integrate over each interval.
#
    e_num = (n - 1) // 2

    for e in range(0, e_num):

        l = 2 * e
        xl = x[l]
        ul = u[l]

        m = 2 * e + 1
        xm = x[m]
        um = u[m]

        r = 2 * e + 2
        xr = x[r]
        ur = u[r]

        for q in range(0, quad_num):

            xq = ((1.0 - abscissa[q]) * xl
                  + (1.0 + abscissa[q]) * xr) \
                / 2.0

            wq = weight[q] * (xr - xl) / 2.0

            vxl = (1.0 / (xl - xm)) \
                * ((xq - xr) / (xl - xr)) \
                + ((xq - xm) / (xl - xm)) \
                * (1.0 / (xl - xr))

            vxm = (1.0 / (xm - xl)) \
                * ((xq - xr) / (xm - xr)) \
                + ((xq - xl) / (xm - xl)) \
                * (1.0 / (xm - xr))

            vxr = (1.0 / (xr - xl)) \
                * ((xq - xm) / (xr - xm)) \
                + ((xq - xl) / (xr - xl)) \
                * (1.0 / (xr - xm))

            uxq = u[l] * vxl + u[m] * vxm + u[r] * vxr
#
#  The piecewise quadratic derivative is a constant in the interval.
#
            uxq = (ur - ul) / (xr - xl)

            exq = exact_ux(xq)

            h1s = h1s + wq * (uxq - exq) ** 2

    h1s = np.sqrt(h1s)

    return h1s


if (__name__ == '__main__'):
    a = 0.5
    xb = 0.2
    fem1d_bvp_quadratic_test00()
    l2_error_quadratic_test()


