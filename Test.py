#%%

import torch
from torch import atan
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as pltcon
from scipy.interpolate import lagrange
from scipy.misc import derivative

from sympy import symbols, integrate, sinh, E, diff
import sympy as sp
from scipy.special import roots_legendre
from pmethod import fem1d_pmethod

def exact_fn(x):

    value = (1 - x) * (np.arctan(a * (x - xb)) + np.arctan(a*xb))
    # value = x * ( 1 - x ) * np.exp ( x )
    return value


def rhs_fn(x):  # PDE

    B = x-xb
    value = -2*(a+a**3*B*(B-x+1))/(a**2*B**2+1)**2
    # value = -x * ( x + 3 ) * np.exp ( x )
    return value

if __name__=="__main__":

  a = 0.5
  xb = 0.2
  # err, u, up = fem1d_linear(exact_fn, rhs_fn, 6)
  err, u, up = fem1d_pmethod(exact_fn, rhs_fn, s=51,  p=5)
  # print(err)
