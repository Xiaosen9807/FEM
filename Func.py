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


def f_test(x):
    return -2*E/(E**2 - 1)*sinh(x) + x
    
def fn_test(x):
    return 0.14588*x*(1-x)+0.16279*x**2*(1-x)
    

def G_integrate(u, x, N=10, scale=(0, 1)):
    N = N  # 取3个样本点
    a = scale[0]  # 积分上下限
    b = scale[1]
    x, w = roots_legendre(N)
    #print(x)
    x = torch.Tensor(x)
    w = torch.Tensor(w)

    xp = x*(b-a)/2+(b+a)/2
    wp = w*(b-a)/2

    s = 0
    for i in range(N):
        s += wp[i]*u.evalf(subs={'x':xp[i]})
    return s.evalf()



def error(u, un, x):
    du = diff(u, x)
    B = G_integrate(du**2+u**2, x)
    A = (1/2*B)**0.5
    dut = diff(un-u, x)
    Bt = G_integrate(dut**2+(un-u)**2, x)
    At = (1/2*Bt)**0.5
    return float(At.evalf()/A.evalf())
