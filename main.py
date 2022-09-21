# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from hashlib import new
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from scipy.misc import derivative

from sympy import symbols
from sympy import *
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



xjm1, xj, xjp1 = sympy.symbols(['x_{j-1}', 'x_{j}', 'x_{j+1}'])
x, h, eps = sympy.symbols(['x', 'h', '\epsilon'])
rho = lambda x: sympy.sin(sympy.pi * x)+1
rho_ = f(x, 0.5, 0.8)

A = - sympy.integrate(rho(x) * (x - xjm1)/h, (x, xjm1, xj))
B = - sympy.integrate(rho(x) * (xjp1 - x)/h, (x, xj, xjp1))

result = sympy.simplify(A + B)
result_func = lambdify(x, result)
#new_result = result_func(x_j)
print(result)
print(type(rho(x)))
#print(new_result)

#print(error(u, result))


