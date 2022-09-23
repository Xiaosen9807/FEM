#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from scipy.misc import derivative

from sympy import symbols


from sympy import *
import sympy as sp

from h_linear import fem1d_linear
from h_quadratic import fem1d_quadratic
from Func import *


def f(x):
    return (1 - x) * (sp.atan(a * (x - xb)) + sp.atan(a*xb))


def d2f(x):
    B = x-xb
    return -2*(a+a**3*B*(B-x+1))/(a**2*B**2+1)**2


a = 0.5
xb = 0.2
err_l, up_l, u_l = fem1d_linear(f, d2f, 6) 
err_q, up_q, u_q = fem1d_quadratic(f, d2f, 7)


print(err_l)
plt.plot(err_l, label='error of of linear elements')

print(err_q)
plt.plot(err_q, label='error of of quadratic elements')
plt.legend()


x_n = len(u_l)
  
x_lo = 0.0
x_hi = 1
x = np.linspace(x_lo, x_hi, x_n)

# U is an approximation to sin(x).

# u = np.zeros(x_n)
# for i in range(0, x_n):
#     u[i] = np.sin(x[i])

e1 = l2_error_quadratic(x_n, x, u_l, f)

print('  %2d  %g' % (x_n, e1))
