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
    #return x * ( 1 - x ) * np.exp ( x )
    return (1 - x) * (sp.atan(a * (x - xb)) + sp.atan(a*xb))


def d2f(x):
    B = x-xb
    #return -x * ( x + 3 ) * np.exp ( x )
    return -2*(a+a**3*B*(B-x+1))/(a**2*B**2+1)**2



a = 0.5
xb = 0.2
err_l_tot=[]
err_q_tot=[]
x_data = 2**np.linspace(1, 5, 5)
for i in x_data:
    
    err_l, up_l, u_l = fem1d_linear(f, d2f, int(i)) 
    err_q, up_q, u_q = fem1d_quadratic(f, d2f, int(i)+1)
    
    
    # print(err_l)
    # #plt.plot(err_l, label='error of of linear elements')
    
    # print(err_q)
    #plt.plot(err_q, label='error of of quadratic elements')
    #plt.legend()
    
    
    x_n = len(up_l)
    #print(len(up_l))
    x_lo = 0.0
    x_hi = 1
    x = np.linspace(x_lo, x_hi, x_n)
    
    # U is an approximation to sin(x).


    e_l = l2_error_quadratic(x_n, x, up_l, f)
    e_q = l2_error_quadratic(x_n, x, up_q, f)
    err_l_tot.append(sp.log(e_l))
    err_q_tot.append(sp.log(e_q))
    # err_l_tot.append(e_l)
    # err_q_tot.append(e_q)
    print('  %2d  %g' % (x_n, e_l))
    print('  %2d  %g' % (x_n, e_q))
    
print(len(x_data))
plt.scatter(np.log(x_data), err_l_tot, label='Linear Error')
plt.scatter(np.log(x_data), err_q_tot, label='Quadratic Error')
plt.legend()
plt.show()
print(err_l_tot)
print(err_q_tot)


# %%
