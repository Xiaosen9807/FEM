# %%

import numpy as np
from sympy import *
import sympy as sp
from scipy.special import roots_legendre
from sympy import *
import scipy.linalg as la
import matplotlib.pyplot as plt
import copy


def Legendre(x=np.linspace(-1, 1, 100), p=5):

    if p == 0:
        return 1
    elif p == 1:
        return x

    else:
        return ((2*p-1)*x*Legendre(x, p-1)+(1-p)*Legendre(x, p-2))/p


# def Hierarchical(x=np.linspace(0, 1, 3), p=5):
#     phi = []
#     dphi = []
#     phi.append((1-x)/2)
#     phi.append((1+x)/2)
#     dphi.append(-1/2)
#     dphi.append(1/2)

#     for j in range(2, p+1):
#         phi.append(1/np.sqrt(4*j-2)*(Legendre(x, j)-Legendre(x, j-2)))
#         dphi.append(np.sqrt(j-1/2)*(Legendre(x, j-1)))

#     return phi, dphi
def Hierarchical(x=np.linspace(0, 1, 3), p=5):
    # if p == 0:
    #     return (1-x)/2, np.zeros_like(x)-0.5
    # elif p == 1:
    #     return (1+x)/2, np.zeros_like(x)+0.5
    if p == 0:
        return (1-x)/2, np.zeros_like(x)-0.5
    elif p == 1:
        return (1+x)/2, np.zeros_like(x)+0.5
    else:
        return 1/np.sqrt(4*p-2)*(Legendre(x, p)-Legendre(x, p-2)), np.sqrt(p-1/2)*(Legendre(x, p-1))


def G_integrate(u, x, N=3, scale=(0, 1)):
    N = N  # 取3个样本点
    a = scale[0]  # 积分上下限
    b = scale[1]
    x, w = roots_legendre(N)
    # print(x)

    xp = x*(b-a)/2+(b+a)/2
    wp = w*(b-a)/2

    s = 0
    for i in range(N):
        s += wp[i]*u(xp[i])
    return s


def fem1d_pmethod(f, d2f, s=50, p=5):
    #
    #  Define the mesh, N+1 points between A and B.
    #  These will be X[0] through X[N].
    #
    a = 0.0
    b = 1.0
    x = np.linspace(a, b, s+1)
#
#  Set a 3 point quadrature rule on the reference interval [-1,1].
#
    q_num = 10

    xg, wg = roots_legendre(q_num)

#
#  Compute the system matrix A and right hand side RHS.
#
    # A = np.zeros((n + 1, n + 1))
    # rhs = np.zeros(n + 1)
    A = np.zeros((p + 1, p + 1))
    rhs = np.zeros(p + 1)

    # for iq = 1 : quad_num
    # x = quad_x(iq);
    # for i = 0 : np
    #   [ phii, phiix ] = phi ( alpha, beta, i, np, x );
    #   for j = 0 : np

    #     [ phij, phijx ] = phi ( alpha, beta, j, np, x );

    #     bij = pp ( x, problem ) * phiix * phijx ...
    #         + qq ( x, problem ) * phii * phij;

    #     b(i+1,j+1) = b(i+1,j+1) + bij * quad_w(iq);

    for q in range(q_num):
        # xl = x[0]
        # xr = x[-1]
        # xq = ((1.0 - xg[q]) * xl + (1.0 + xg[q]) * xr) / 2.0
        # wq = wg[q] * (xr - xl) / 2.0
        xq = xg[q]
        wq = wg[q]

        for i in range(p+1):
            phii, phiix = Hierarchical(xq, i)
            rhs[i] += wq * phii * d2f(xq)

            for j in range(p+1):
                phij, phijx = Hierarchical(xq, j)
                Aij = phii * phij + phiix * phijx
                A[i, j] += Aij * wq

    # A[0, 0] = 1.0
    # A[0, 1:p+1] = 0.0
    # A[p, p] = 1.0
    # A[p, 0:p] = 0.0

    # rhs[0] = f(x[0])
    # rhs[p] = f(x[-1])

    print(A)

    print('rhs', rhs)
#  Solve the linear system.
#
    u_ = la.solve(A, -rhs)
    print(u_)
    # plt.plot(u_)
    phi = []
    for i in range(p+1):
        phi.append(Hierarchical(x, i)[0])
        #plt.scatter(x, Hierarchical(x, i)[0])
    phi = np.array(phi)

    u = np.dot(phi.T, u_)
    print(u)

    # plt.show()


#  Evaluate the exact solution at the nodes.

    uex = np.zeros(p + 1)
    for i in range(0, p + 1):
        uex[i] = f(x[i])
    err = []
    for i in range(0, p + 1):
        err.append(abs(uex[i] - u[i]))
        #print('  %4d  %14.6g  %14.6g  %14.6g' % (i, u[i], uex[i], err))

#  Plot the computed solution and the exact solution.
#  Evaluate the exact solution at enough points that the curve will look smooth.

    npp = 51
    xp = np.linspace(a, b, npp)
    up = np.zeros(npp)
    for i in range(0, npp):
        up[i] = f(xp[i])

    # plt.plot(x, u, 'bo-', xp, up, 'r.')
    filename = 'fem1d.png'
    plt.savefig(filename)
    # plt.show()

    # plt.figure()
    plt.plot(x, u, 'bo-', label='u')
    plt.plot(xp, up, 'r.', label='up')
    plt.title('P-method')
    plt.legend()
    plt.show()
    # print(xp)

    return err, u, up


def exact_fn(x, a=0.5, xb=0.2):

    value = (1 - x) * (np.arctan(a * (x - xb)) + np.arctan(a*xb))
    # value = x * ( 1 - x ) * np.exp ( x )
    return value


def rhs_fn(x, a=0.5, xb=0.2):  # PDE

    B = x-xb
    value = -2*(a+a**3*B*(B-x+1))/(a**2*B**2+1)**2
    # value = -x * ( x + 3 ) * np.exp ( x )
    return value


if __name__ == "__main__":
    x = symbols("x")
    # print(Hierarchical(x))
    a = 0.5
    xb = 0.2
    # err, u, up = fem1d_linear(exact_fn, rhs_fn, 6)
    err, u, up = fem1d_pmethod(exact_fn, rhs_fn, s=5, p=3)
    # print(err)
