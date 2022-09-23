# %%

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import time


def fem1d_linear(n=10):
    #
    #  Define the mesh, N+1 points between A and B.
    #  These will be X[0] through X[N].
    #
    a = 0.0
    b = 1.0
    x = np.linspace(a, b, n + 1)
#
#  Set a 3 point quadrature rule on the reference interval [0,1].
#
    ng = 3

    xg = np.array((
        0.112701665379258311482073460022,
        0.5,
        0.887298334620741688517926539978))

    wg = np.array((
        5.0 / 18.0,
        8.0 / 18.0,
        5.0 / 18.0))
#
#  Compute the system matrix A and right hand side RHS.
#
    A = np.zeros((n + 1, n + 1))
    rhs = np.zeros(n + 1)
#
#  Look at element E: (0, 1, 2, ..., N-1).
#
    for e in range(0, n):

        xl = x[e]
        xr = x[e+1]
#
#  Consider quadrature point Q: (0, 1, 2 ) in element E.
#
        for q in range(0, ng):
            #
            #  Map XG and WG from [0,1] to
            #      XQ and QQ in [XL,XR].
            #
            xq = xl + xg[q] * (xr - xl)
            wq = wg[q] * (xr - xl)
#
#  Consider the I-th test function PHI(I,X) and its derivative PHI'(I,X).
#
            for i_local in range(0, 2):
                i = i_local + e

                if (i_local == 0):
                    phii = (xq - xr) / (xl - xr)
                    phiip = 1.0 / (xl - xr)
                else:
                    phii = (xq - xl) / (xr - xl)
                    phiip = 1.0 / (xr - xl)

                rhs[i] = rhs[i] + wq * phii * rhs_fn(xq)
#
#  Consider the J-th basis function PHI(J,X) and its derivative PHI'(J,X).
#  (It turns out we don't need PHI for this particular problem, only PHI')
#
                for j_local in range(0, 2):
                    j = j_local + e

                    if (j_local == 0):
                        phijp = 1.0 / (xl - xr)
                    else:
                        phijp = 1.0 / (xr - xl)

                    A[i][j] = A[i][j] + wq * phiip * phijp
#
#  Modify the linear system to enforce the left boundary condition.
#
    A[0, 0] = 1.0
    A[0, 1:n+1] = 0.0
    rhs[0] = exact_fn(x[0])
#
#  Modify the linear system to enforce the right boundary condition.
#
    A[n, n] = 1.0
    A[n, 0:n] = 0.0
    rhs[n] = exact_fn(x[n])
    # plt.plot(rhs)
    # plt.show()

    print(A)
    print('_-----------------------\n', rhs)
#  Solve the linear system.
#
    u = la.solve(A, rhs)
#
#  Evaluate the exact solution at the nodes.
#
    uex = np.zeros(n + 1)
    for i in range(0, n + 1):
        uex[i] = exact_fn(x[i])
    err = []
    for i in range(0, n + 1):
        err.append(abs(uex[i] - u[i]))
        # print('  %4d  %14.6g  %14.6g  %14.6g' % (i, u[i], uex[i], err))
#
#  Plot the computed solution and the exact solution.
#  Evaluate the exact solution at enough points that the curve will look smooth.
#
    npp = 51
    xp = np.linspace(a, b, npp)
    up = np.zeros(npp)
    for i in range(0, npp):
        up[i] = exact_fn(xp[i])

    #plt.plot(x, u, 'bo-', xp, up, 'r.')
    filename = 'fem1d.png'
    plt.savefig(filename)
    # plt.show()

    # plt.figure()
    plt.plot(x, u, 'bo-', label='true')
    plt.plot(xp, up, 'r.', label='simu')
    plt.legend()
    plt.show()
    print(xp)

    return err, u, up


def exact_fn(x):

    value = (1 - x) * (np.arctan(a * (x - xb)) + np.arctan(a*xb))
    return value


def rhs_fn(x): # PDE

    B = x-xb
    value = 2*(a+a**3*B*(B-x+1))/(a**2*B**2+1)**2
    return value


if __name__ == '__main__':
    a = 0.5
    xb = 0.2
    err, u, up = fem1d_linear(5)
    print(err)
# %%
