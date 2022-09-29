#%%
import numpy as np
from sympy import *
import sympy as sp
from scipy.special import roots_legendre
from sympy import *
import scipy.linalg as la
import matplotlib.pyplot as plt


def Legendre(x=np.linspace(-1, 1, 100), p=5 ):

    if p == 0:
        return 1
    elif p == 1:
        return x
    
    else:
      return ((2*p-1)*x*Legendre(x, p-1)+(1-p)*Legendre(x, p-2))/p

def Hierarchical(x=np.linspace(0, 1, 3), p=5):
    phi = []
    dphi = []
    phi.append((1-x)/2)
    phi.append((1+x)/2)
    dphi.append(-1/2)
    dphi.append(1/2)
    
    for j in range(2, p+1):
        phi.append(1/np.sqrt(4*j-2)*(Legendre(x, j)-Legendre(x, j-2)))
        dphi.append(np.sqrt(j-1/2)*(Legendre(x, j-1)))
        
    return phi, dphi
  
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
    n = s
    a = 0.0
    b = 1.0
    x = np.linspace(a, b, n + 1)
#
#  Set a 3 point quadrature rule on the reference interval [-1,1].
#
    q_num = 3

    xg = np.array((
        -0.774596669241483377035853079956,
        0.0,
        0.774596669241483377035853079956))

    wg = np.array((
        5.0 / 9.0,
        8.0 / 9.0,
        5.0 / 9.0))

#
#  Compute the system matrix A and right hand side RHS.
#
    A = np.zeros((n + 1, n + 1))
    rhs = np.zeros(n + 1)
#
#  Look at element E: (0, 1, 2, ..., N-1).
#
    for p_ in range(2, p+1):
        # if p_==0:
        #     phi.append((1-x)/2)
        #     phi.append((1+x)/2)
        #     dphi.append(np.array([-1/2, -1/2, -1/2]))
        #     dphi.append(np.array([1/2, 1/2, 1/2]))
        for e in range(0, n):

            xl = x[e]
            xr = x[e+1]
    #
    #  Consider quadrature point Q: (0, 1, 2 ) in element E.
    #
            for q in range(0, q_num):
                #
                #  Map XG and WG from [-1,1] to
                #      XQ and QQ in [XL,XR].
                #
                # xq = xl + xg[q] * (xr - xl)
                # wq = wg[q] * (xr - xl)
                # xq = xl + (xg[q] + 1.0) * (xr - xl) / 2.0
                # wq = wg[q] * (xr - xl) / 2.0
                xq = ((1.0 - xg[q]) * xl + (1.0 + xg[q]) * xr) / 2.0
                wq = wg[q] * (xr - xl) / 2.0
    #
    #  Consider the I-th test function PHI(I,X) and its derivative PHI'(I,X).
    #
                for i_local in range(0, 2):
                    i = i_local + e
                    if p_ == 0:
                        phii = (1-xq)/2
                        phiip = -0.5
                    elif p_ == 1:
                        phii = (1+xq)/2
                        phiip = 0.5
                    else:
                        phii = 1/np.sqrt(4*p_-2) * \
                            (Legendre(xq, p_)-Legendre(xq, p_-2))
                        phiip = np.sqrt(p_-1/2)*(Legendre(xq, p_-1))

                        # if (i_local == 0):
                        #     phii = (xq - xr) / (xl - xr)
                        #     phiip = 1 / (xl - xr)
                        # else:
                        #     phii = (xq - xl) / (xr - xl)
                        #     phiip = 1 / (xr - xl)

                    rhs[i] = rhs[i] + wq * phii * d2f(xq)
    #
    #  Consider the J-th basis function PHI(J,X) and its derivative PHI'(J,X).
    #  (It turns out we don't need PHI for this particular problem, only PHI')
    #
                    for j_local in range(0, 2):
                        j = j_local + e

                        # if (j_local == 0):
                        #     phijp = 1 / (xl - xr)
                        #     #print(phiip)
                        # else:
                        #     phijp = 1 / (xr - xl)
                        if p_ == 0:
                            phijp = -0.5
                        elif p_ == 1:
                            phijp = 0.5
                        else:
                            phijp = np.sqrt(p_-1/2)*(Legendre(xq, p_-1))

                        A[i][j] = A[i][j] + wq * phiip * phijp

#
#  Modify the linear system to enforce the left boundary condition.
#
    A[0, 0] = 1.0
    A[0, 1:n+1] = 0.0
    rhs[0] = f(x[0])
#
#  Modify the linear system to enforce the right boundary condition.
#
    A[n, n] = 1.0
    A[n, 0:n] = 0.0
    rhs[n] = f(x[n])

    # print(A)
    # print('_-----------------------\n', rhs)
#  Solve the linear system.
#
    u = la.solve(A, -rhs)
#
#  Evaluate the exact solution at the nodes.
#
    uex = np.zeros(n + 1)
    for i in range(0, n + 1):
        uex[i] = f(x[i])
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
        up[i] = f(xp[i])

    # plt.plot(x, u, 'bo-', xp, up, 'r.')
    filename = 'fem1d.png'
    plt.savefig(filename)
    # plt.show()

    # plt.figure()
    plt.plot(x, u, 'bo-', label='u')
    plt.plot(xp, up, 'r.', label='up')
    plt.title('h_linear')
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

if __name__=="__main__":
  x = symbols("x")
  print(Hierarchical(x))
  a = 0.5
  xb = 0.2
  # err, u, up = fem1d_linear(exact_fn, rhs_fn, 6)
  err, u, up = fem1d_pmethod(exact_fn, rhs_fn, p=5)
  # print(err)

  

