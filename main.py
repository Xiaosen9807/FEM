
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
