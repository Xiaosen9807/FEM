#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from scipy.misc import derivative

from sympy import symbols


from sympy import *
import sympy as sp
from scipy.special import roots_legendre
from h_linear import fem1d_linear
from h_quadratic import fem1d_quadratic


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
plt.plot(err_l, label='errof of linear elements')

print(err_q)
plt.plot(err_q, label='errof of quadratic elements')
plt.legend()
