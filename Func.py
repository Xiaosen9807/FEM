#%%

import sympy  # 符号运算库
import torch
from torch import atan
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from scipy.misc import derivative

from sympy import symbols, integrate, sinh, E, diff
import sympy as sp
from scipy.special import roots_legendre


def f_test(x):
    return -2*E/(E**2 - 1)*sinh(x) + x


def fn_test(x):
    return 0.14588*x*(1-x)+0.16279*x**2*(1-x)


def G_integrate(u, x, N=3, scale=(0, 1)):
    N = N  # 取3个样本点
    a = scale[0]  # 积分上下限
    b = scale[1]
    x, w = roots_legendre(N)
    # print(x)
    x = torch.Tensor(x)
    w = torch.Tensor(w)

    xp = x*(b-a)/2+(b+a)/2
    wp = w*(b-a)/2

    s = 0
    for i in range(N):
        s += wp[i]*u.evalf(subs={'x': xp[i]})
    return s.evalf()


def cal_U(u):
    x = symbols('x')
    du = diff(u, x)
    #B = integrate(du**2+u**2, (x, 0, 1))
    B = G_integrate(du**2+u**2, x)
    U = (1/2*B)**0.5
    return U.evalf()
# def error(u, un):
#     x = symbols('x')
#     du = diff(u, x)
#     B = G_integrate(du**2+u**2, x)
#     A = (1/2*B)**0.5
#     dut = diff(u-un, x)
#     Bt = G_integrate(dut**2+(u-un)**2, x)
#     At = (1/2*Bt)**0.5
#     return float((At.evalf()/A.evalf()))


def error(u, un):

    return cal_U(u-un)/cal_U(u)


def lagrange(x, y, num_points, x_test):
    # 所有的基函数值，每个元素代表一个基函数的值
    l = np.zeros(shape=(num_points, ))

    # 计算第k个基函数的值
    for k in range(num_points):
        # 乘法时必须先有一个值
        # 由于l[k]肯定会被至少乘n次，所以可以取1
        l[k] = 1
        # 计算第k个基函数中第k_个项（每一项：分子除以分母）
        for k_ in range(num_points):
            # 这里没搞清楚，书中公式上没有对k=k_时，即分母为0进行说明
            # 有些资料上显示k是不等于k_的
            if k != k_:
                # 基函数需要通过连乘得到
                l[k] = l[k]*(x_test-x[k_])/(x[k]-x[k_])
            else:
                pass
    # 计算当前需要预测的x_test对应的y_test值
    L = 0
    for i in range(num_points):
        # 求所有基函数值的和
        L += y[i]*l[i]
    return L


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

    e2 = 0.0
    #
    #  Define a 2 point Gauss-Legendre quadrature rule on [-1,+1].
    #
    quad_num = 3
    abscissa, weight= roots_legendre(quad_num)
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
    
    #print('e2', e2)

    e2 = e2**0.5

    return e2


if __name__ == '__main__':
    x_n = 11

    x_lo = 0.0
    x_hi = np.pi
    x = np.linspace(x_lo, x_hi, x_n)

    # U is an approximation to sin(x).

    u = np.zeros(x_n)
    for i in range(0, x_n):
        u[i] = np.sin(x[i])

    e1 = l2_error_quadratic(x_n, x, u, np.sin)

    print('  %2d  %g' % (x_n, e1))
