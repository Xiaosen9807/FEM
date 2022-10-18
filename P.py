import numpy as np
    
def fem1d_pmethod(): 
    #*****************************************************************************80
    
    ## MAIN is the main program for FEM1D_PMETHOD.
    
    #  Discussion:
    
    #    FEM1D_PMETHOD implements the P-version of the finite element method.
    
    #    Program to solve the one dimensional problem:
    
    #      - d/dX (P dU/dX) + Q U  =  F
    
    #    by the finite-element method using a sequence of polynomials
#    which satisfy the boundary conditions and are orthogonal
#    with respect to the inner product:
    
    #      (U,V)  =  Integral (-1 to 1) P U' V' + Q U V dx
    
    #    Here U is an unknown scalar function of X defined on the
#    interval [-1,1], and P, Q and F are given functions of X.
    
    #    The boundary values are U(-1) = U(1)=0.
    
    #    Sample problem #1:
    
    #      U=1-x^4,        P=1, Q=1, F=0+12.0*x^2-x^4
    
    #    Sample problem #2:
    
    #      U=cos(0.5*pi*x), P=1, Q=0, F=0.25*pi*pi*cos(0.5*pi*x)
    
    #    The program should be able to get the exact solution for
#    the first problem, using NP = 2.
    
    #  Licensing:
    
    #    This code is distributed under the GNU LGPL license.
    
    #  Modified:
    
    #    03 November 2006
    
    #  Author:
    
    #    Original FORTRAN77 version by Max Gunzburger, Teresa Hodge.
#    MATLAB version by John Burkardt.
    
    #  Local Parameters:
    
    #    Local, real A(0:NP), the squares of the norms of the
#    basis functions.
    
    #    Local, real ALPHA(NP), BETA(NP), the recurrence coefficients.
#    for the basis functions.
    
    #    Local, real F(1:NP+1).
#    F contains the basis function coefficients that form the
#    representation of the solution U.  That is,
#      U(X)  =  SUM (I=0 to NP) F(I+1) * BASIS(I)(X)
#    where "BASIS(I)(X)" means the I-th basis function
#    evaluated at the point X.
    
    #    Local, integer NP.
#    The highest degree polynomial to use.
    
    #    Local, integer NPRINT.
#    The number of points at which the computed solution
#    should be printed out at the end of the computation.
    
    #    Local, integer PROBLEM, indicates the problem being solved.
#    1, U=1-x^4, P=1, Q=1, F=0+12.0*x^2-x^4.
#    2, U=cos(0.5*pi*x), P=1, Q=0, F=0.25*pi*pi*cos(0.5*pi*x).
    
    #    Local, integer QUAD_NUM, the order of the quadrature rule.
    
    #    Local, real QUAD_W(QUAD_NUM), the quadrature weights.
    
    #    Local, real QUAD_X(QUAD_NUM), the quadrature abscissas.
    
    print('\n' % ())
    print('FEM1D_PMETHOD\n' % ())
    print('  MATLAB version\n' % ())
    np = 2
    quad_num = 10
    nprint = 10
    problem = 2
    print('\n' % ())
    print('  Solve the two-point boundary value problem\n' % ())
    print('\n' % ())
    print('  - d/dX (P dU/dX) + Q U  =  F\n' % ())
    print('\n' % ())
    print('  on the interval [-1,1], with\n' % ())
    print('  U(-1) = U(1) = 0.\n' % ())
    print('\n' % ())
    print('  The P method is used, which represents U as\n' % ())
    print('  a weighted sum of orthogonal polynomials.\n' % ())
    print('\n' % ())
    print('  Highest degree polynomial to use is %d\n' % (np))
    print('  Number of points to be used for output = %d\n' % (nprint))
    if (problem == 1):
        print('\n' % ())
        print('  Problem #1:\n' % ())
        print('  U=1-x^4,\n' % ())
        print('  P=1,\n' % ())
        print('  Q=1,\n' % ())
        print('  F=1 + 12 * x^2 - x^4\n' % ())
    else:
        if (problem == 2):
            print('\n' % ())
            print('  Problem #2:\n' % ())
            print('  U=cos(0.5*pi*x),\n' % ())
            print('  P=1,\n' % ())
            print('  Q=0,\n' % ())
            print('  F=0.25*pi*pi*cos(0.5*pi*x)\n' % ())
    
    
    #  Get quadrature abscissas and weights for interval [-1,1].
    
    quad_w,quad_x = quad(quad_num)
    
    #  Compute the constants for the recurrence relationship
#  that defines the basis functions.
    
    a,alpha,beta = alpbet(np,problem,quad_num,quad_w,quad_x)
    
    #  Test the orthogonality of the basis functions.
    
    ortho(a,alpha,beta,np,problem,quad_num,quad_w,quad_x)
    
    #  Solve for the solution of the problem, in terms of coefficients
#  of the basis functions.
    
    f = sol(a,alpha,beta,np,problem,quad_num,quad_w,quad_x)
    
    #  Print out the solution, evaluated at each of the NPRINT points.
    
    solution_print(alpha,beta,f,np,nprint)
    
    #  Compare the computed and exact solutions.
    
    exact(alpha,beta,f,np,nprint,problem,quad_num,quad_w,quad_x)
    
    #  Terminate.
    
    print('\n' % ())
    print('FEM1D_PMETHOD\n' % ())
    print('  Normal end of execution.\n' % ())
    print('\n' % ())
    return f
    return
    
    
def alpbet(np = None,problem = None,quad_num = None,quad_w = None,quad_x = None): 
    #*****************************************************************************80
    
    ## ALPBET calculates the coefficients in the recurrence relationship.
    
    #  Discussion:
    
    #    ALPHA and BETA are the coefficients in the three
#    term recurrence relation for the orthogonal basis functions
#    on [-1,1].
    
    #    The routine also calculates A, the square of the norm of each basis
#    function.
    
    #  Licensing:
    
    #    This code is distributed under the GNU LGPL license.
    
    #  Modified:
    
    #    03 November 2006
    
    #  Author:
    
    #    Original FORTRAN77 version by Max Gunzburger, Teresa Hodge
#    MATLAB version by John Burkardt
    
    #  Parameters:
    
    #    Output, real A(1:NP+1), the squares of the norms of the
#    basis functions.
    
    #    Output, real ALPHA(NP), BETA(NP), the recurrence coefficients.
#    for the basis functions.
    
    #    Input, integer NP.
#    The highest degree polynomial to use.
    
    #    Input, integer PROBLEM, indicates the problem being solved.
#    1, U=1-x^4, P=1, Q=1, F=0+12.0*x^2-x^4.
#    2, U=cos(0.5*pi*x), P=1, Q=0, F=0.25*pi*pi*cos(0.5*pi*x).
    
    #    Input, integer QUAD_NUM, the order of the quadrature rule.
    
    #    Input, real QUAD_W(QUAD_NUM), the quadrature weights.
    
    #    Input, real QUAD_X(QUAD_NUM), the quadrature abscissas.
    
    ss = 0.0
    su = 0.0
    beta = np.zeros_like
    for iq in np.arange(1,quad_num+1).reshape(-1):
        x = quad_x(iq)
        s = 4.0 * pp(x,problem) * x * x + qq(x,problem) * (0 - x * x) ** 2
        u = 2.0 * pp(x,problem) * x * (3.0 * x * x - 0) + x * qq(x,problem) * (0 - x * x) ** 2
        ss = ss + s * quad_w(iq)
        su = su + u * quad_w(iq)
    
    beta[1] = 0.0
    a[1] = ss
    alpha[1] = su / ss
    for i in np.arange(2,np + 1+1).reshape(-1):
        ss = 0.0
        su = 0.0
        sv = 0.0
        for iq in np.arange(1,quad_num+1).reshape(-1):
            x = quad_x(iq)
            #  Three term recurrence for Q and Q'.
            qm1 = 0.0
            q = 0
            qm1x = 0.0
            qx = 0.0
            for k in np.arange(1,i - 1+1).reshape(-1):
                qm2 = qm1
                qm1 = q
                q = (x - alpha(k)) * qm1 - beta(k) * qm2
                qm2x = qm1x
                qm1x = qx
                qx = qm1 + (x - alpha(k)) * qm1x - beta(k) * qm2x
            t = 0 - x * x
            #  The basis function PHI = ( 1 - x^2 ) * q.
            #     s = pp * ( phi(i) )' * ( phi(i) )' + qq * phi(i) * phi(i)
            s = pp(x,problem) * (t * qx - 2.0 * x * q) ** 2 + qq(x,problem) * (t * q) ** 2
            #     u = pp * ( x * phi(i) )' * phi(i)' + qq * x * phi(i) * phi(i)
            u = pp(x,problem) * (x * t * qx + (0 - 3.0 * x * x) * q) * (t * qx - 2.0 * x * q) + x * qq(x,problem) * (t * q) ** 2
            #     v = pp * ( x * phi(i) )' * phi(i-1) + qq * x * phi(i) * phi(i-1)
            v = pp(x,problem) * (x * t * qx + (0 - 3.0 * x * x) * q) * (t * qm1x - 2.0 * x * qm1) + x * qq(x,problem) * t * t * q * qm1
            #  SS(i) = <   phi(i), phi(i)   > = integral ( S )
#  SU(i) = < x phi(i), phi(i)   > = integral ( U )
#  SV(i) = < x phi(i), phi(i-1) > = integral ( V )
            ss = ss + s * quad_w(iq)
            su = su + u * quad_w(iq)
            sv = sv + v * quad_w(iq)
        a[i] = ss
        #  ALPHA(i) = SU(i) / SS(i)
#  BETA(i)  = SV(i) / SS(i-1)
        if (i <= np):
            alpha[i] = su / ss
            beta[i] = sv / a(i - 1)
    
    return a,alpha,beta
    return a,alpha,beta
    
    
def exact(alpha = None,beta = None,f = None,np = None,nprint = None,problem = None,quad_num = None,quad_w = None,quad_x = None): 
    #*****************************************************************************80
    
    ## EXACT compares the computed and exact solutions.
    
    #  Licensing:
    
    #    This code is distributed under the GNU LGPL license.
    
    #  Modified:
    
    #    03 November 2006
    
    #  Author:
    
    #    Original FORTRAN77 version by Max Gunzburger, Teresa Hodge
#    MATLAB version by John Burkardt
    
    #  Parameters:
    
    #    Input, real ALPHA(NP), BETA(NP), the recurrence coefficients.
#    for the basis functions.
    
    #    Input, real F(1:NP+1).
#    F contains the basis function coefficients that form the
#    representation of the solution U.  That is,
#      U(X)  =  SUM (I=0 to NP) F(I+1) * BASIS(I)(X)
#    where "BASIS(I)(X)" means the I-th basis function
#    evaluated at the point X.
    
    #    Input, integer NP.
#    The highest degree polynomial to use.
    
    #    Input, integer NPRINT.
#    The number of points at which the computed solution
#    should be printed out at the end of the computation.
    
    #    Input, integer PROBLEM, indicates the problem being solved.
#    1, U=1-x^4, P=1, Q=1, F=0+12.0*x^2-x^4.
#    2, U=cos(0.5*pi*x), P=1, Q=0, F=0.25*pi*pi*cos(0.5*pi*x).
    
    #    Input, integer QUAD_NUM, the order of the quadrature rule.
    
    #    Input, real QUAD_W(QUAD_NUM), the quadrature weights.
    
    #    Input, real QUAD_X(QUAD_NUM), the quadrature abscissas.
    
    nsub = 10
    print('\n' % ())
    print('  Comparison of computed and exact solutions:\n' % ())
    print('\n' % ())
    print('    X        U computed    U exact     Difference\n' % ())
    print('\n' % ())
    for i in np.arange(0,nprint+1).reshape(-1):
        x = (2 * i - nprint) / nprint
        ue = uex(x,problem)
        up = 0.0
        for j in np.arange(0,np+1).reshape(-1):
            phii,phiix = phi(alpha,beta,j,np,x)
            up = up + phii * f(j + 1)
        print('  %8f  %12f  %12f  %12f\n' % (x,up,ue,ue - up))
    
    
    #  Compute the big L2 error.
    
    big_l2 = 0.0
    for i in np.arange(1,nsub+1).reshape(-1):
        xl = (2 * i - nsub - 1) / nsub
        xr = (2 * i - nsub) / nsub
        for j in np.arange(1,quad_num+1).reshape(-1):
            x = (xl * (0 - quad_x(j)) + xr * (0 + quad_x(j))) / 2.0
            up = 0.0
            for k in np.arange(0,np+1).reshape(-1):
                phii,phiix = phi(alpha,beta,k,np,x)
                up = up + phii * f(k + 1)
            big_l2 = big_l2 + (up - uex(x,problem)) ** 2 * quad_w(j) * (xr - xl) / 2.0
    
    big_l2 = np.sqrt(big_l2)
    print('\n' % ())
    print('  Big L2 error = %f\n' % (big_l2))
    return
    return
    
    
def ff(x = None,problem = None): 
    #*****************************************************************************80
    
    ## FF evaluates the right hand side function F(X) at any point X.
    
    #  Licensing:
    
    #    This code is distributed under the GNU LGPL license.
    
    #  Modified:
    
    #    03 November 2006
    
    #  Author:
    
    #    Original FORTRAN77 version by Max Gunzburger, Teresa Hodge
#    MATLAB version by John Burkardt
    
    #  Parameters:
    
    #    Input, real X, the evaluation point.
    
    #    Input, integer PROBLEM, indicates the problem being solved.
#    1, U=1-x^4, P=1, Q=1, F=0+12.0*x^2-x^4.
#    2, U=cos(0.5*pi*x), P=1, Q=0, F=0.25*pi*pi*cos(0.5*pi*x).
    
    #    Output, real VALUE, the value of F(X).
    
    
    #  Test problem 1
    
    if (problem == 1):
        value = 0 + 12.0 * x ** 2 - x ** 4
        #  Test problem 2
    else:
        if (problem == 2):
            value = 0.25 * np.pi ** 2 * np.cos(0.5 * np.pi * x)
    
    return value
    return value
    
    
def ortho(a = None,alpha = None,beta = None,np = None,problem = None,quad_num = None,quad_w = None,quad_x = None): 
    #*****************************************************************************80
    
    ## ORTHO tests the basis functions for orthogonality.
    
    #  Licensing:
    
    #    This code is distributed under the GNU LGPL license.
    
    #  Modified:
    
    #    03 November 2006
    
    #  Author:
    
    #    Original FORTRAN77 version by Max Gunzburger, Teresa Hodge
#    MATLAB version by John Burkardt
    
    #  Parameters:
    
    #    Input, real A(1:NP+1), the squares of the norms of the
#    basis functions.
    
    #    Input, real ALPHA(NP), BETA(NP), the recurrence coefficients.
#    for the basis functions.
    
    #    Input, integer NP.
#    The highest degree polynomial to use.
    
    #    Input, integer PROBLEM, indicates the problem being solved.
#    1, U=1-x^4, P=1, Q=1, F=0+12.0*x^2-x^4.
#    2, U=cos(0.5*pi*x), P=1, Q=0, F=0.25*pi*pi*cos(0.5*pi*x).
    
    #    Input, integer QUAD_NUM, the order of the quadrature rule.
    
    #    Input, real QUAD_W(QUAD_NUM), the quadrature weights.
    
    #    Input, real QUAD_X(QUAD_NUM), the quadrature abscissas.
    
    
    #  Zero out the B array, so we can start summing up the dot products.
    
    b=np.zeros  
    
    #  Approximate the integral of the product of basis function
#  I and basis function J over the interval [-1,1].
    
    #  We expect to get zero, except when I and J are equal,
#  when we should get A(I).
    
    for iq in np.arange(1,quad_num+1).reshape(-1):
        x = quad_x(iq)
        for i in np.arange(0,np+1).reshape(-1):
            phii,phiix = phi(alpha,beta,i,np,x)
            for j in np.arange(0,np+1).reshape(-1):
                phij,phijx = phi(alpha,beta,j,np,x)
                bij = pp(x,problem) * phiix * phijx + qq(x,problem) * phii * phij
                b[i + 1,j + 1] = b(i + 1,j + 1) + bij * quad_w(iq)
    
    
    #  Print out the results of the test.
    
    print('\n' % ())
    print('  Basis function orthogonality test:\n' % ())
    print('\n' % ())
    print('   i   j     b(i,j)/a(i)\n' % ())
    print('\n' % ())
    for i in np.arange(0,np+1).reshape(-1):
        print('\n' % ())
        for j in np.arange(0,np+1).reshape(-1):
            print('  %4d  %4d  %12f\n' % (i,j,b(i + 1,j + 1) / a(i + 1)))
    
    return
    return
    
    
def phi(alpha = None,beta = None,i = None,np = None,x = None): 
    #*****************************************************************************80
    
    ## PHI evaluates the I-th basis function at the point X.
    
    #  Licensing:
    
    #    This code is distributed under the GNU LGPL license.
    
    #  Modified:
    
    #    03 November 2006
    
    #  Author:
    
    #    Original FORTRAN77 version by Max Gunzburger, Teresa Hodge
#    MATLAB version by John Burkardt
    
    #  Parameters:
    
    #    Input, real ALPHA(NP), BETA(NP), the recurrence coefficients.
#    for the basis functions.
    
    #    Input, integer I, the index of the basis function.
    
    #    Input, integer NP, the highest degree polynomial to use.
    
    #    Input, real X, the evaluation point.
    
    #    Output, real PHII, PHIIX, the value of the basis
#    function and its derivative.
    
    if (0):
        qm1 = 0.0
        q = 0
        for j in np.arange(1,i+1).reshape(-1):
            qm2 = qm1
            qm1 = q
            q = (x - alpha(j)) * qm1 - beta(j) * qm2
        phii = (0 - x * x) * q
        qm1x = 0.0
        qx = 0.0
        for j in np.arange(1,i+1).reshape(-1):
            qm2x = qm1x
            qm1x = qx
            qx = qm1 + (x - alpha(j)) * qm1x - beta(j) * qm2x
        phiix = (0 - x * x) * qx - 2.0 * x * q
    else:
        qm1 = 0.0
        q = 0
        qm1x = 0.0
        qx = 0.0
        for j in np.arange(1,i+1).reshape(-1):
            qm2 = qm1
            qm1 = q
            qm2x = qm1x
            qm1x = qx
            t = x - alpha(j)
            q = t * qm1 - beta(j) * qm2
            qx = qm1 + t * qm1x - beta(j) * qm2x
        t = 0 - x * x
        phii = t * q
        phiix = t * qx - 2.0 * x * q
    
    return phii,phiix
    return phii,phiix
    
    
def pp(x = None,problem = None): 
    #*****************************************************************************80
    
    ## PP returns the value of the coefficient function P(X).
    
    #  Licensing:
    
    #    This code is distributed under the GNU LGPL license.
    
    #  Modified:
    
    #    03 November 2006
    
    #  Author:
    
    #    Original FORTRAN77 version by Max Gunzburger, Teresa Hodge
#    MATLAB version by John Burkardt
    
    #  Parameters:
    
    #    Input, real X, the evaluation point.
    
    #    Input, integer PROBLEM, indicates the problem being solved.
#    1, U=1-x^4, P=1, Q=1, F=0+12.0*x^2-x^4.
#    2, U=cos(0.5*pi*x), P=1, Q=0, F=0.25*pi*pi*cos(0.5*pi*x).
    
    #    Output, real VALUE, the value of P(X).
    
    
    #  Test problem 1
    
    if (problem == 1):
        value = 0
        #  Test problem 2
    else:
        if (problem == 2):
            value = 0
    
    return value
    return value
    
    
def qq(x = None,problem = None): 
    #*****************************************************************************80
    
    ## QQ returns the value of the coefficient function Q(X).
    
    #  Licensing:
    
    #    This code is distributed under the GNU LGPL license.
    
    #  Modified:
    
    #    03 November 2006
    
    #  Author:
    
    #    Original FORTRAN77 version by Max Gunzburger, Teresa Hodge
#    MATLAB version by John Burkardt
    
    #  Parameters:
    
    #    Input, real X, the evaluation point.
    
    #    Input, integer PROBLEM, indicates the problem being solved.
#    1, U=1-x^4, P=1, Q=1, F=0+12.0*x^2-x^4.
#    2, U=cos(0.5*pi*x), P=1, Q=0, F=0.25*pi*pi*cos(0.5*pi*x).
    
    #    Output, real VALUE, the value of Q(X).
    
    
    #  Test problem 1
    
    if (problem == 1):
        value = 0
        #  Test problem 2
    else:
        if (problem == 2):
            value = 0.0
    
    return value
    return value
    
    
def quad(quad_num = None): 
    #*****************************************************************************80
    
    ## QUAD returns the abscissas and weights for gaussian quadrature on [-1,1].
    
    #  Licensing:
    
    #    This code is distributed under the GNU LGPL license.
    
    #  Modified:
    
    #    03 November 2006
    
    #  Author:
    
    #    Original FORTRAN77 version by Max Gunzburger, Teresa Hodge
#    MATLAB version by John Burkardt
    
    #  Parameters:
    
    #    Input, integer QUAD_NUM, the order of the quadrature rule.
    
    #    Output, real QUAD_W(QUAD_NUM), the quadrature weights.
    
    #    Output, real QUAD_X(QUAD_NUM), the quadrature abscissas.
    
    
    #  Quadrature points
    
    quad_x[np.arange[1,10+1]] = np.array([[- 0.973906528517172],[- 0.865063366688985],[- 0.679409568299024],[- 0.433395394129247],[- 0.148874338981631],[0.148874338981631],[0.433395394129247],[0.679409568299024],[0.865063366688985],[0.973906528517172]])
    
    #  Weights
    
    quad_w[np.arange[1,10+1]] = np.array([[0.066671344308688],[0.149451349150581],[0.219086362515982],[0.269266719309996],[0.295524224714753],[0.295524224714753],[0.269266719309996],[0.219086362515982],[0.149451349150581],[0.066671344308688]])
    return quad_w,quad_x
    return quad_w,quad_x
    
    
def r8vec_print(n = None,a = None,title = None): 
    #*****************************************************************************80
    
    ## R8VEC_PRINT prints a real vector.
    
    #  Licensing:
    
    #    This code is distributed under the GNU LGPL license.
    
    #  Modified:
    
    #    25 January 2004
    
    #  Author:
    
    #    John Burkardt
    
    #  Parameters:
    
    #    Input, integer N, the dimension of the vector.
    
    #    Input, real A(N), the vector to be printed.
    
    #    Input, string TITLE, a title.
    
    print('\n' % ())
    print('%s\n' % (title))
    print('\n' % ())
    for i in np.arange(1,n+1).reshape(-1):
        print('%6d: %12g\n' % (i,a(i)))
    
    return
    return
    
    
def sol(a = None,alpha = None,beta = None,np = None,problem = None,quad_num = None,quad_w = None,quad_x = None): 
    #*****************************************************************************80
    
    ## SOL solves a linear system for the finite element coefficients.
    
    #  Licensing:
    
    #    This code is distributed under the GNU LGPL license.
    
    #  Modified:
    
    #    03 November 2006
    
    #  Author:
    
    #    Original FORTRAN77 version by Max Gunzburger, Teresa Hodge
#    MATLAB version by John Burkardt
    
    #  Parameters:
    
    #    Input, real A(1:NP+1), the squares of the norms of the
#    basis functions.
    
    #    Input, real ALPHA(NP), BETA(NP), the recurrence coefficients.
#    for the basis functions.
    
    #    Input, integer NP.
#    The highest degree polynomial to use.
    
    #    Input, integer PROBLEM, indicates the problem being solved.
#    1, U=1-x^4, P=1, Q=1, F=0+12.0*x^2-x^4.
#    2, U=cos(0.5*pi*x), P=1, Q=0, F=0.25*pi*pi*cos(0.5*pi*x).
    
    #    Input, integer QUAD_NUM, the order of the quadrature rule.
    
    #    Input, real QUAD_W(QUAD_NUM), the quadrature weights.
    
    #    Input, real QUAD_X(QUAD_NUM), the quadrature abscissas.
    
    #    Output, real F(1:NP+1).
#    F contains the basis function coefficients that form the
#    representation of the solution U.  That is,
#      U(X)  =  SUM (I=0 to NP) F(I+1) * BASIS(I)(X)
#    where "BASIS(I)(X)" means the I-th basis function
#    evaluated at the point X.
    
    f[np.arange[1,np + 1+1]] = 0.0
    for iq in np.arange(1,quad_num+1).reshape(-1):
        x = quad_x(iq)
        t = ff(x,problem) * quad_w(iq)
        for i in np.arange(0,np+1).reshape(-1):
            phii,phiix = phi(alpha,beta,i,np,x)
            f[i + 1] = f(i + 1) + phii * t
    
    f[np.arange[1,np + 1+1]] = f(np.arange(1,np + 1+1)) / a(np.arange(1,np + 1+1))
    return f
    return f
    
    
def solution_print(alpha = None,beta = None,f = None,np = None,nprint = None): 
    #*****************************************************************************80
    
    ## SOLUTION_PRINT prints out the computed solution.
    
    #  Licensing:
    
    #    This code is distributed under the GNU LGPL license.
    
    #  Modified:
    
    #    03 November 2006
    
    #  Author:
    
    #    Original FORTRAN77 version by Max Gunzburger, Teresa Hodge.
#    MATLAB version by John Burkardt.
    
    #  Parameters:
    
    #    Input, real ALPHA(NP), BETA(NP), the recurrence coefficients.
#    for the basis functions.
    
    #    Input, real F(1:NP+1).
#    F contains the basis function coefficients that form the
#    representation of the solution U.  That is,
#      U(X)  =  SUM (I=0 to NP) F(I+1) * BASIS(I)(X)
#    where "BASIS(I)(X)" means the I-th basis function
#    evaluated at the point X.
    
    #    Input, integer NP.
#    The highest degree polynomial to use.
    
    #    Input, integer NPRINT.
#    The number of points at which the computed solution
#    should be printed out at the end of the computation.
    
    print('\n' % ())
    print('  Representation of solution:\n' % ())
    print('\n' % ())
    print('  Basis function coefficients:\n' % ())
    print('\n' % ())
    for i in np.arange(0,np+1).reshape(-1):
        print('  %4d  %12f\n' % (i,f(i + 1)))
    
    print('\n' % ())
    print('\n' % ())
    print('       X     Approximate Solution\n' % ())
    print('\n' % ())
    for ip in np.arange(0,nprint+1).reshape(-1):
        x = (2 * ip - nprint) / nprint
        up = 0.0
        for i in np.arange(0,np+1).reshape(-1):
            phii,phiix = phi(alpha,beta,i,np,x)
            up = up + phii * f(i + 1)
        print('  %12f  %12f\n' % (x,up))
    
    print('\n' % ())
    return
    return

    
    
def uex(x = None,problem = None): 
    #*****************************************************************************80
    
    ## UEX returns the value of the exact solution at a point X.
    
    #  Licensing:
    
    #    This code is distributed under the GNU LGPL license.
    
    #  Modified:
    
    #    03 November 2006
    
    #  Author:
    
    #    Original FORTRAN77 version by Max Gunzburger, Teresa Hodge
#    MATLAB version by John Burkardt
    
    #  Parameters:
    
    #    Input, real X, the evaluation point.
    
    #    Input, integer PROBLEM, indicates the problem being solved.
#    1, U=1-x^4, P=1, Q=1, F=0+12.0*x^2-x^4.
#    2, U=cos(0.5*pi*x), P=1, Q=0, F=0.25*pi*pi*cos(0.5*pi*x).
    
    #    Output, real VALUE, the exact value of U(X).
    
    
    #  Test problem 1
    
    if (problem == 1):
        value = 0 - x ** 4
        #  Test problem 2
    else:
        if (problem == 2):
            value = np.cos(0.5 * np.pi * x)
    
    return value

fem1d_pmethod()