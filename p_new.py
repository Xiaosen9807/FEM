#%%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from ipywidgets import interactive
import ipywidgets as widgets
from ExternalFunctions import Lobatto as Lo

# for dark theme aficionados
#plt.style.use('dark_background')


# # parameters of the problem
rho0 = 1; c0 = 1;
rho_0 = 1; c0 = 1; # normalized fluid properties
beta = 1; Vn = 1/(rho_0*c0); # boundary conditions

def initiate(DuctLength,NrOfElem,Order):
  from ExternalFunctions import Mesh1D, CreateDofs
  NrOfNodes, Coord, Element = Mesh1D(DuctLength,NrOfElem)
  NrOfDofs, DofNode, DofElement = CreateDofs(NrOfNodes,NrOfElem,Element,Order)
  return NrOfNodes, Coord, Element, NrOfDofs, DofNode, DofElement

def assemble_matrix(NrOfDofs,NrOfElem,Order,Coord,Element,DofElement,omega,c0):
  from ExternalFunctions import MassAndStiffness_1D
  Matrix = np.zeros((NrOfDofs,NrOfDofs), dtype=np.complex128)
  for iElem in np.arange(0,NrOfElem):
      # call the function returning the mass and stifness element matrices
      Ke, Me = MassAndStiffness_1D(iElem, Order, Coord, Element)
      ElemDofs = (DofElement[:,iElem]).astype(int)
      # assemble - [side note "irregular slice" requires np.ix_ in python]
      Matrix[np.ix_(ElemDofs,ElemDofs)] += Ke - (omega/c0)**2*Me
  return Matrix

def assemble_impedance(Matrix,DofNode,NrOfNodes,omega,c0,beta):
  # now apply impedance boundary condition at last node
  Matrix[DofNode[NrOfNodes-1],DofNode[NrOfNodes-1]] += 1j*omega/c0*beta # Equation
  return Matrix

def assemble_rhs(NrOfDofs,DofNode,omega,rho0,Vn):
  # and the velocity at first node
  Rhs = np.zeros((NrOfDofs,1), dtype=np.complex128)
  Rhs[DofNode[0]] = 1j*omega*rho0*Vn
  return Rhs

def solve(Matrix, Rhs):
  # solve the sparse system of equations 
  Sol = np.linalg.solve(Matrix, Rhs) 
  return Sol

def compute_model(omega, Order, NrOfElem, DuctLength):
  from ExternalFunctions import GetSolutionOnSubgrid
  h = DuctLength/NrOfElem # mesh size
  d_lambda = 2*math.pi/(omega/c0*h)*Order # nr of dofs per wavelength
  # first create the mesh and the Dofs list
  NrOfNodes, Coord, Element, NrOfDofs, DofNode, DofElement = initiate(DuctLength,NrOfElem,Order)
  # then assemble the matrix
  Matrix = assemble_matrix(NrOfDofs,NrOfElem,Order,Coord,Element,DofElement,omega,c0)
  # then assemble the rhs 
  Matrix = assemble_impedance(Matrix,DofNode,NrOfNodes,omega,c0,beta)
  # now apply impedance boundary condition at last node
  Rhs = assemble_rhs(NrOfDofs,DofNode,omega,rho0,Vn)
  # solve the sparse system of equations 
  Sol = solve(Matrix, Rhs)
  # compute the solution on a subgrid 
  Lambda = 2*math.pi/(omega/c0); NrOfWavesOnDomain = DuctLength/Lambda
  x_sub, u_h_sub = GetSolutionOnSubgrid(Sol, Order, Coord, Element, NrOfElem, DofElement, NrOfWavesOnDomain)
  # exact solution on subgrid 
  u_exact_sub = np.exp(-1j*omega/c0*x_sub)
  return x_sub, u_h_sub, u_exact_sub, NrOfDofs, d_lambda, Sol, Coord, Element
    
    
def plot_result(x_sub, u_h_sub, u_exact_sub, NrOfDofs, d_lambda, NrOfElem, Coord, Element, Order):
  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4), tight_layout=True)
  ax.plot(x_sub, np.real(u_h_sub), label="$Re(u_h)$",linewidth=2)
  ax.plot(x_sub, np.real(u_exact_sub), label="$Re(u_{exact})$",linewidth=2)
  for iElem in np.arange(0,NrOfElem):
      x_nodes = Coord[Element[0: 2, iElem].astype(int)]
      xi = np.linspace(-1,1,20); N1 = (1-xi)/2; N2 = (1+xi)/2; 
      x = N1*x_nodes[0] + N2*x_nodes[1]
      ax.plot(x, Lo(xi,Order),linestyle=":",color="gray",linewidth=1)  
  ax.legend(loc="best",fontsize=16)
  ax.set(xlabel=r'$x$', ylabel=r'real part',
         title='NrOfDofs = '+str(NrOfDofs) + ', ($d_\lambda$=%1.4g' %d_lambda +' Dofs per wavelength)')
  plt.show()

def compute_interactive(omega, Order, NrOfElem, DuctLength):
  x_sub, u_h_sub, u_exact_sub, NrOfDofs, d_lambda, Sol, Coord, Element = compute_model(omega, Order, NrOfElem, DuctLength)
  plot_result(x_sub, u_h_sub, u_exact_sub, NrOfDofs, d_lambda, NrOfElem, Coord, Element, Order)
  E2 = np.linalg.norm(u_h_sub - u_exact_sub)/np.linalg.norm(u_exact_sub)*100
  print('-' *100 +'\n'+' Numerical error is %1.4g' %E2 + ' % \n' +'-' *100)
  
  

compute_interactive(10, 3, 20, 1)