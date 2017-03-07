#-------------------------------------------------------------------------------
# Name:        RBF Design Matrix
# Purpose:      Create RBF Design Matrix H[p,m]
#               Constructs a design matrix from the input training points,
#               the hidden unit positions, sizes and function types. Allows
#               Gaussian, Cauchy, multiquadric and inverse multiquadric type
#               functions, and an optional bias unit.
#
# Author:      Mohammad
#
# Created:     04/05/2016
# Copyright:   (c) ISI 2016
# Licence:     ISI
#-------------------------------------------------------------------------------

import os, sys
import numpy as np

from Tools_RBF import *

mypath    = os.path.dirname(os.path.realpath(sys.argv[0]))

# ---------------------------
#   Matrix Dimension Guide
# ---------------------------
#   n: Size of Inputs               [x1 .. xn]
#   m: Size of Bases                [h1 .. hm]
#   p: Number of Training Samples   {X,y}p
#   k: Number of Test     Samples   {X,y}k
#
#
#   H  [p,m]
#   W  [m,n]
#   A  [m,m]
#   Y  [p,n] or [p,k]
#   P  [p,p]
#
#   HA [p,m]
#   F  [p,M>>m]
#   l  [1,M]  l[q]
#   U  [m,m]
#   PY [p,n]



#===============================================================================
#           RBF Design Matrix H
#===============================================================================
def rbfDesign(X, C, R, options=None):
    """
    # H = rbfDesign(X, C, R, options)
    #
    # PDF: Sections 3.1, A.3 - Radial Basis Function
    # Gets the design matrix from the input data, center positions
    # and radii factors.
    #
    #        | h1(X1) .. hm(X1) |
    #   H =  | .            .   |
    #        | h1(Xp) .. hm(Xp) |
    #
    #   n: Number of Inputs             X=[x1 .. xn]
    #   m: Number of Basis              [h1 .. hm]
    #   p: Number of Training Samples   {X,y}p
    #
    #   h(X) = Theta( (X-C)' R^-1 (X-C) )
    #       X:Training, C: Centers, R: Metric (R=r^2I Euclidean Metric)
    #
    #   hj(X) = exp( - (x-c)^2 / r^2 )   Gaussian
    #
    # Input
    #       X       Input training data [X1 .. Xp]    (n-by-p)
    #       C       List of centers [C1 .. Cm]        (n-by-m)
    #       R	    Scale factors                     (scalar, n-vector, or n-by-n matrix)
    #	    options	Specifying basis function type    (string, list of strings )
    #
    #               'g' = Gaussian              (0) Theta(z)=     e^-z
    #               'c' = Cauchy                (1) Theta(z)= (1+z)^-1.0
    #               'm' = Multiquadric          (2) Theta(z)= (1+z)^+0.5
    #               'i' = Inverse Multiquadric  (3) Theta(z)= (1+z)^-0.5
    #
    #               'b' = bias unit required
    #
    #
    # Output
    #       H       Design matrix                       (p-by-m)

    """
    # set default type /bias
    type = 0    # Gaussian
    bias = 0    # No Bias

    # get options
    # ---------------------------
    if options is not None:
    	for option in options:
            option = option.lower()

            if option == 'g':
                # gaussian (0)
    			type = 0
            elif option == 'c':
                # cauchy (1)
    			type = 1
            elif option == 'm':
                # multiquadric (2)
    			type = 2
            elif option == 'i':
                # inverse multiquadric (3)
    			type = 3
            elif option.lower() == 'b':
    			bias = 1
            else:
    			raise AssertionError('rbfDesign: illegal option')



    # preliminary sizing
    # ---------------------------
    #   n: Number of Inputs             X=[x1 .. xn]
    #   m: Number of Basis              [h1 .. hm]
    #   p: Number of Training Samples   {X,y}p
    [n, p]   = X.shape
    [n1, m]  = C.shape


    if not isinstance(R, (np.ndarray, list)):
        rr,rc = 1,1
    else:
        if np.array(R).ndim==1:
            R = np.atleast_2d(np.array(R))    # Row Vector
        [rr, rc] = R.shape

    if n != n1:
    	raise AssertionError('rbfDesign: mismatched X, C')


    # determine scaling type
    # ---------------------------
    """
    #   X[n,p]=[X1 .. Xp]   C[n,m]=[C1 .. Cm]
    #
    #   SCALING_TYPE (R Col Vector or Matrix)
    #   1   same radius for each center               R[1,1]
  	#	2	same diagonal metric for each center      R[n,1] or R[1,n].T
    #   3	same metric for each center               R[n,n]
  	#   4	different radius for each center          R[m,1] or R[1,m].T
    #   5	different diagonal metric for each center R[n,m] or R[m,n].T
    """

    if rr == 1 and rc == 1:
    	SCALING_TYPE = 1	      # same radius for each center

    elif rr == 1:
    	if rc == n:
    		SCALING_TYPE = 2	 # same diagonal metric for each center
    		R = R.conj().T
    	elif rc == m:
    		SCALING_TYPE = 4	 # different radius for each center
    		R = R.conj().T
    	else:
    		error('rbfDesign: mismatched C and row vector R')

    elif rc == 1:
    	if rr == n:
    		SCALING_TYPE = 2	# same diagonal metric for each center
    	elif rr == m:
    		SCALING_TYPE = 4	# different radius for each center
    	else:
    		error('rbfDesign: mismatched C and row vector R')

    elif rr == n:
    	if rc == n:
    		SCALING_TYPE = 3	# same metric for each center
    		IR = numpy.linalg.inv(R)
    	elif rc == m:
    		SCALING_TYPE = 5	# different diagonal metric for each center
    	else:
    		error('rbfDesign: mismatched C and matrix R')

    elif rc == n:
    	if rr == m:
    		SCALING_TYPE = 5	# different diagonal metric for each center
    		R = R.conj().T
    	else:
    		error('rbfDesign: mismatched C and matrix R')

    else:
    	error('rbfDesign: wrong sized R')


    # start constructing H
    # ---------------------------
    """
    #   X[n,p] = [X1 .. Xp]     Input training data
    #   C[n,m] = [C1 .. Cm]     List of centers
    #   R[m,1][n,1][n,n],[n,m]	Scale factors (Col Vector or Matrix)
    #
    #   D[n,p] = X-C            Difference
    #   H[p,m]                  Design Matrix
    #
    #        | h1(X1) .. hm(X1) |
    #   H =  | .            .   |
    #        | h1(Xp) .. hm(Xp) |
    """
    H = np.zeros((p, m))

    for j in range(0,m):                                                        ## j in [1..m]
    	# get p difference vectors for this center
    	D = X - dupCol(C[:,j], p)                                               # D[n,p] =[n,p]-[n,1 *p]

    	# do metric calculation
    	if SCALING_TYPE == 1:
            # R[1,1] same radius for each center
    		s = diagProduct(D.conj().T,D) / R**2                                # s[p,1] =Diag([p,n][n,p])

    	elif SCALING_TYPE == 2:
            # R[n,1] same diagonal metric for each center
    		DR = D / dupCol(R, p)
    		s = diagProduct(DR.conj().T,DR)

    	elif SCALING_TYPE == 3:
            # R[n,n] same full metric for each center
    		DR = np.dot(IR,D);
    		s = diagProduct(DR.conj().T,DR)

    	elif SCALING_TYPE == 4:
            # R[m,1] different radius for each center
    		s = diagProduct(D.conj().T,D) / R[j,0]**2

        else:
            # R[n,m] different diagonal metric for each center
    		DR = D / dupCol(R[:,j], p)
    		s = diagProduct(DR.conj().T,DR)


    	# apply basis function
        #   s[p,1] --> h[p,1]
        # ---------------------------
    	if type == 0:		           # Gaussian (default)
    		h = np.exp(-s)

    	elif type == 1:	               # Cauchy
    		h = 1 / (s + 1)

    	elif type == 2:	               # multiquadric
    		h = np.sqrt(s + 1)

    	elif type == 3:                # inverse multiquadric
    		h = 1 / np.sqrt(s + 1);

    	# insert result in H
    	H[:, j] = h.ravel()            # h[p,1], H[:, j]=[p,]  use h.ravel or H[;,j:j+1]


    # add bias unit
    # ---------------------------
    if bias:
	   H = np.hstack(( H, np.ones((p, 1)) ))

    return H


# ==============================================================================
#                              RBF Design Example
# ==============================================================================
def _example():
    """
    #
    # Example for rbfDesign manual pages.
    # Recreates stuff from famous Broomhead & Lowe paper.
    #

    #
    # Prepare the figure.
    #

    figure(1)
    pos = get(1, 'Position');
    set(1, ...
      'Position', [pos(1) pos(2) 400 400], ...
      'NumberTitle', 'off', ...
      'Name', 'rbfDesignExample', ...
      'PaperType', 'a4letter', ...
      'InvertHardCopy', 'on', ...
      'PaperPosition', [0.5 0.5 4 4])

    #
    # Get training set.
    X = [[0; 0] [0; 1] [1; 1] [1; 0]];
    y = [0 1 0 1]';

    #
    # Set up the radial basis function network with four
    # multiquadric centers (same positions as inputs) and
    # unit radii.
    C = X;
    r = 1;
    H = rbfDesign(X, C, r, 'm');

    #
    # Train the network by solving for the least square
    # weights using the normal equation.
    w = inv(H' * H) * H' * y;

    #
    # Sanity check.
    (H * w)'



    # Reproduce figure 2 of Broomhead and Lowe.
    #
    d = 100;
    x = linspace(-1, 2, d);
    [X1, X2] = meshgrid(x);
    Xt = [X1(:) X2(:)]';
    Ht = rbfDesign(Xt, C, r, 'm');
    yt = Ht * w;
    Yt = zeros(d,d);
    Yt(:) = yt;
    hold off
    contour(x, x, Yt, [-.18 -.16 0 .28 .71 1 1.57])
    set(gca, 'XTick', [-1 0 1 2])
    set(gca, 'YTick', [-1 0 1 2])
    hold on
    plot([0 0 1 1 0], [0 1 1 0 0], 'w')

    """
    pass
# ==============================================================================
#                              MAIN RUN PART
# ==============================================================================
if __name__ == '__main__':
    pass
