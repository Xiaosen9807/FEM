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

# Set the model
eps_0 = 8.8541878176*10e-12 #F/m
eps_r = 1.0
eps = eps_0*eps_r
rho_0 = 10e-8 # C/m**3
V_0 = 1.0 #Volt

d = 0.08 #m
number_of_elements = 4

l=d/number_of_elements

# After elimination of the equation system using 
# Dirichlet boundary condition we have less equations...
K = np.zeros((number_of_elements-1,
              number_of_elements-1))
for i in range(number_of_elements-1):
    for j in range(number_of_elements-1):
        if i == j: K[i,j]=2
        if abs(i-j) == 1:
            K[i,j] = -1
print(K)

# Still we need all values of the potential
V = np.zeros(number_of_elements + 1)
V[0] = V_0

# Set the right-hand side
f = np.ones(number_of_elements-1)*(-(l**2)*rho_0)/(eps)
f[0] += V_0

# Solve the system
V[1:-1] = np.linalg.solve(K,f)
print(V)

#Plot it with analytic solution.
x_fem = np.linspace(0.0, d, num=number_of_elements+1)
evaluate = (V_0 + rho_0*x**2/(2*varepsilon_0*varepsilon_r)
            + x*(-V_0/d - d*rho_0/(2*varepsilon_0*varepsilon_r)))
x_vec = np.linspace(0.0, d, num=number_of_elements*10+1)
y_vec = np.array([ evaluate.subs({x:value}) for value in x_vec]) 
fig, ax = plt.subplots()
ax.plot(x_vec, y_vec, x_fem,V);
ax.set_xlim(0,0.08);