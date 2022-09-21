
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from scipy.misc import derivative

from sympy import symbols
import sympy as sp
from scipy.special import roots_legendre

from Func import *


def f(x, a=0.5, xb=0.8):
    return (1 - x) * (sp.atan(a * (x - xb)) + sp.atan(a*xb))

    
x = symbols('x')
a = 0.5
xb = 0.8
u = f_test(x)
un = fn_test(x)

print(error(u, un, x))



x = np.linspace(0, 2*np.pi, 10, endpoint=True)  # 0到2*pi，等分十个，最后一个值包含
y = np.sin(x)
"""print(x)
print(y)"""
lag_interp = LagrangeInterpolation(x=x, y=y)
lag_interp.fit_interp()
