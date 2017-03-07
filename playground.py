#-------------------------------------------------------------------------------
# Name:        RBF Playground (Examples)
# Purpose:
#
#
#
# Author:      Mohammad
#
# Created:     04/05/2016
# Copyright:   (c) ISI 2016
# Licence:     ISI
#-------------------------------------------------------------------------------
import os, sys
import numpy as np
import pylab as plt
import scipy.io as sio
import tkFileDialog

from Tools_RBF          import *
from RBFdesign          import *
from FwSelection        import *
from RidgeRegression    import *
mypath    = os.path.dirname(os.path.realpath(sys.argv[0]))


#===============================================================================
#   Matrix Dimension Guide
#===============================================================================
# ---------------------------

# ---------------------------
#   n: Size of Inputs               [x1 .. xn]
#   m: Size of Bases                [h1 .. hm]
#   p: Number of Training Samples   {X,y}p
#   k: Number of Test     Samples   {X,y}k
#
#
#   Xi [n,p]
#   Yi^[p,1]
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
#           MAIN
#===============================================================================
NEW = False

if not NEW:
    ##path_MAT = tkFileDialog.askopenfilename(initialdir=mypath,title="MATLAB Data File",filetypes=[("MATLAB Data",'*.mat'),("File",'*.*')])
    path_MAT = os.path.join(mypath,'introExample.mat')

    data = sio.loadmat(path_MAT)

    x   = data['x']
    y   = data['y']
    n,p = x.shape
    m   = p

else:
    # Basis Set
    #============================
    sigma   = 0.2
    p       = 50
    n       = 1
    m       = p

    x = 1 -np.random.rand(1,p)                                  # [n=1,p]
    y = np.sin(10 * x).T + sigma * np.random.randn(p,1)         # [p ,1]


# Training Set
#============================
pt = 500
xt = np.linspace(0,1,pt)[:,np.newaxis].T                    # [n=1,pt]
yt = np.sin (10 * xt).T                                     # [pt ,1]
print "Base     X,y:  ", x.shape, y.shape
print "Training X,y:  ",  xt.shape, yt.shape
print "-----------------------------"



# Design Matrix
#============================
c = x
r = 0.1
H = rbfDesign(x,c,r,'c')                                    # H[p,m=p]


# Least Square (Find Optimal Weigths)
#============================
w = np.linalg.inv(np.dot(H.T,H)).dot(H.T).dot(y)            # w[m=p,n=1] =[p,p][p,p][p,p][p,1]
Ht = rbfDesign(xt,c,r,'c')                                  # Ht[pt,m=p]
ft = np.dot(Ht,w)                                           # ft[pt,1]  =[pt,p][m=p,n=1]



# Forward Selection
#============================
out     = forwardSelect(H, y, '-t BIC -v')
subset  = out[0]        # subset, H, l, U, A, w, P
Hs = H[:,subset]
w  = np.linalg.inv(Hs.T.dot(Hs)).dot(Hs.T).dot(y)
ft = Ht[:,subset].dot(w)
print "Forward Select MSE = ", 1./pt * (ft-yt).T.dot(ft-yt)
print "-----------------------------"


# Global Ridge
#============================
lamda = 1e-4
w       = np.linalg.inv(np.dot(H.T,H)+lamda*np.eye(m)).dot(H.T).dot(y)        # w[m=p,n=1] =[p,p][p,p][p,p][p,1]
ft      = Ht.dot(w)
print "Pre Global Ridge MSE = ", 1./pt * (ft-yt).T.dot(ft-yt)
print "-----------------------------"

out     = globalRidge(H,y, 0.1, 'BIC -v')
[lamda, e, L, E] = out
print "Lambda = ", lamda
w       = np.linalg.inv(np.dot(H.T,H)+lamda*np.eye(m)).dot(H.T).dot(y)        # w[m=p,n=1] =[p,p][p,p][p,p][p,1]
ft      = Ht.dot(w)
print "Post Global Ridge MSE = ", 1./pt * (ft-yt).T.dot(ft-yt)
print "-----------------------------"



# Local Ridge
#============================
##lamda = 1.0862     #0.016                                                                  # from GeneratorExiGlobal Ridge
[lamdas, V1, H1, A1, W1, P1] = localRidge(H,y,lamda, '-v')

finite = np.where(np.isfinite(lamdas))[0]
print " Finite lamdas = ", finite+1

lamdas = lamdas[finite]
Hl  = H[:,finite]
w   = np.linalg.inv(np.dot(Hl.T,Hl) + np.diag(lamdas)).dot(Hl.T).dot(y)
ft  = Ht[:,finite].dot(w)
print "Local Ridge MSE = ", 1./pt * (ft-yt).T.dot(ft-yt)
print "-----------------------------"




#============================
# Figures
#============================
plt.figure()
plt.ylim([-1.5,1.5])
plt.plot(x.ravel(),y.ravel(),'+r')
plt.plot(xt.ravel(),yt.ravel(),'k')
plt.plot(xt.ravel(),ft.ravel(),'g')


lamdas = np.logspace(-4,2,100)
bics = np.zeros(100)
for i in range(100):
    [e, W, A, P, g] = predictError(H,y,lamdas[i],'BIC')         # out = [e, W, A, P, g]
    bics[i] = e[0]
[e, W, A, P, g] = predictError(H,y,lamda, "BIC")
bic = e[0]
plt.figure()
plt.loglog(lamdas,bics,'m-')
plt.loglog(lamda,bic,'ro')



plt.show()
