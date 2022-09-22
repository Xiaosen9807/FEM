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

print('hello')

print(torch.linspace(1, 10, 10))
print(np.linspace(1, 10, 10))