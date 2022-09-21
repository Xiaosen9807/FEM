import sympy # 符号运算库
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

def error(u, un):
    x = symbols('x')
    du = diff(u, x)
    B = G_integrate(du**2+u**2, x)
    A = (1/2*B)**0.5
    dut = diff(un-u, x)
    Bt = G_integrate(dut**2+(un-u)**2, x)
    At = (1/2*Bt)**0.5
    return float(At.evalf()/A.evalf())

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


class LagrangeInterpolation:
    '''
    拉格朗日插值
    '''

    def __init__(self, x, y):
        '''
        拉格朗日必要参数的初始化，及各健壮性的检测；健壮性：系统在不正常输入情况下仍能表现正常的程度
        :param x: 已知数据x的坐标点
        :param y: 已知数据y的坐标点
        '''
        #构造一下作为类的属性，防止后续进行数组的运算,做一个类型转换
        self.x = np.asarray(x, dtype=np.float)
        self.y = np.asarray(y, dtype=np.float)  # 类型转换，数据结构采用array

        if len(self.x) > 1 and len(self.x) == len(self.y):  # 如果x，y是一个数据点，就不用做了，且个数应相同
            self.n = len(self.x)  # 有多少个数据点，即已知离散数据点的个数
        else:
            raise ValueError("插值数据（x，y）维度不匹配")
        #插值最终结果是一个多项式的形式，可作为一个类的属性由用户进行返回
        self.polynomial = None  # 最终的插值多项式，符号表示
        self.poly_coefficient = None  # 最终插值多项式的系数向量，幂次从高到低
        self.coefficient_order = None  # 对应多项式系数的阶次（所有项中最高次的幂）
        self.y0 = None  # 所求插值点的值，单个值或向量

    def fit_interp(self):  # 拟合
        '''
        核心算法：生成拉格朗日插值多项式
        :return:
        '''
        t = sympy.Symbol("t")  # 定义符号变量，因为参数已经用过x
        self.polynomial = 0.0  # 插值多项式实例化
        for i in range(self.n):  # 由多少个样本点，n个数据生成n-1次
            # 根据每个数据带点，构造插值基函数
            basis_fun = self.y[i]  # 插值基函数
            # 计算每个 yi*li（x）
            for j in range(i):
                basis_fun *= (t-self.x[j]) / (self.x[i]-self.x[j])

            for j in range(i + 1, self.n):
                basis_fun *= (t - self.x[j]) / (self.x[i] - self.x[j])

            # 下面开始累加
            self.polynomial += basis_fun  # 插值多项式累加
        self.polynomial = sympy.expand(self.polynomial)  # 多项式的展开
        print(self.polynomial)

    def cal_interp_x0(self, x0):  # 计算插值的值
        '''
        计算所给定的插值点的值，即插值
        :param x0: 所求插值的x坐标
        :return:
        '''
        pass
