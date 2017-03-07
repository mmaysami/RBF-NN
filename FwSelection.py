#-------------------------------------------------------------------------------
# Name:        RBF Forward Selection
# Purpose:      Does forward subset selection with (optionally) OLS, ridge
#               regression (fixed or variable lambda) and a choice of
#               stopping criteria.
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


"""
    A.dot(B).dot(C)
"""


#===============================================================================
#           Forward Selection
#===============================================================================
def forwardSelect(F, Y, options=None):
    """
    # [subset, H, l, U, A, w, P]  = forwardSelect(F, Y, options)
    #
    # PDF: Sections 7 - Forward Selection Critria (& A.12 - A.14)
    # Regularized orthogonal least squares algorithm.
    # See "Regularization in the Selection of Radial Basis
    # Function centers", 1995, Orr, M.J.L., Neural Computation,
    # 7(3):606-623.
    #
    #   n: Size of Inputs               [x1 .. xn]
    #   m: Size of Bases                [h1 .. hm]
    #   p: Number of Training Samples   {X,y}p
    #   k: Number of Test     Samples   {X,y}k
    #
    # Inputs
    #
    #   F        Design matrix of selectable centers  (p-by-M)
    #   Y	     Output training data                 (p-by-n)
    #   options  Control options                      (string)
    #           -v          Verbose = True
    #           -V          Verbose, Flops = True
    #           -t str #    Termination method, Threshold/MaxAge value (VAR 0<Threshold<1,  Rest MaxAge>=1)
    #           -m     #<1  Max number of regressors allowed in subset, MaxReg value
    #           -g          Global regularization required, Optional initial lambda (#>0)
    #           -r str #    Turn on Global regularization, Optional initial lambda (#>0)
    #           OLS         Turn on Orthogonal Least-Squares
    #
    # Output
    #
    #   subset   Indices of selected columns of F     (1-by-m)
    #   H        Subset of F                          (p-by-m)
    #   l        regularization parameter             (real and non-negative)
    #   U        U the upper triangular tarnsform     (m-by-m)
    #   A        inv(H'*H + l * U' * U)               (m-by-m)
    #   w        A * H' * Y                           (m-by-n)
    #   P        I - H * A * H'                       (p-by-p)
    #
    # The various Termination criteria used are:
    #
    #   VAR  Fraction of Explained Variance
    #   GCV  Generalized Cross Validation
    #   UEV  Unbiased Estimate of Variance
    #   FPE  Final Prediction Error
    #   BIC  Bayesian Information Criterion
    #
    # specified in options by, e.g. 'FPE'.
    """

    # Default options
    # ---------------------------
    Model_Dict  = {
                    'v':'VAR',  #  Fraction of Explained Variance
                    'g':'GCV',
                    'u':'UEV',
                    'f':'FPE',
                    'b':'BIC',
                    'l':'LOO',
                  }

    Verbose     = False
    Flops       = False
    Global      = False
    OLS         = False
    Term        = 'g'
    MaxAge      = 2
    MaxReg      = 0
    ReEst       = 'n'           # No Re-Estimate
    Threshold   = 0.9
    lamda     = 0.1



    # Preliminaries (Assertion)
    # ---------------------------
    #   n: Size of Inputs               [x1 .. xn]
    #   m: Size of Bases                [h1 .. hm]
    #   p: Number of Training Samples   {X,y}p
    #
    #   H[p,m]    Y[p,n]  F[p,M>>m]    P[p,p]
    [ph,M] = F.shape
    [py,n] = Y.shape
    if py != ph:
      raise ValueError('forwardSelect: design and outputs have incompatible dimensions')
    p = py


    # process options
    # ---------------------------
    if options is not None:
        # initialize
        i = 0                                                                   ## i =1
        [arg, i] = getNextArg(options, i)

        # scan through arguments
        while arg != '':
            if arg =='-v':
                # verbose output required
                Verbose = True

            elif arg=='-V':
                # verbose output required with compute cost reporting
                Verbose = True
                Flops   = True

            # ----------------
            elif arg=='-t':
                """ # specify termination criterion (method or number) """
                [arg, i] = getNextArg(options, i)

                # 1. Check for method first
                if arg.lower() in ['var','gcv','uev','fpe','bic','loo']:
                    #  use Fraction of Explained Variance, UEV, FPE, GCV, BIC, and LOO to terminate
                    method_given = True
                    Term = arg[0].lower()
                    # read next argument
                    [arg, ii] = getNextArg(options, i)
                else:
                    # the method wasn't specified, or specified incorrectly
                    method_given = False
                    print('forwardSelect: terminate with VAR, UEV, FPE, GCV, BIC or LOO')
                    raise ValueError('forwardSelect: bad or missing argument for -t option')

                # 2. is a number given?
                try:
                    th = np.float(arg)
                except:
                    th = None

                if th is not None:
                    good_value   = True
                    # a value, threshold or maximum age, is specified
                    if Term == 'v' and (th>0 and th<1):
                        # valid value for threshold methods
                        Threshold = th
                    elif (Term in 'gufbl') and (th>=1):
                        # valid value for maximum age methods
                        MaxAge = np.round(th)
                    else:
                        # Invalid value
                        good_value = False


                    if good_value:
                        # get ready to advance to next arg
                        i = ii
                    else:
                        print('forwardSelect: acceptable termination Threshold/MaxAge values are:')
                        print('  VAR: 0 < Threshold value < 1')
                        print('  UEV, FPE, GCV, BIC or LOO: MaxAge value >= 1')
                        raise ValueError('forwardSelect: bad value for -t option')


            # ----------------
            elif arg=='-m':
                """ # maximum number of regressors allowed in subset (needs value) """
                [arg, i] = getNextArg(options, i)

                # 2. is a number given?
                try:
                    mr = np.float(arg)
                except:
                    mr = None

                if mr is not None:
                    good_value = True
                    MaxReg = np.round(mr)
                    if MaxReg < 1:
                        good_value = False
                else:
                    # the value argument is mandatory
                    good_value = False

                if not good_value:
                    print('forwardSelect: positive maximum size required')
                    raise ValueError('forwardSelect: bad or missing value for -m option')


            # ----------------
            elif arg=='-g':
                """ # global regularization required """
                Global = True

                # is the initial value of lamda given (Optional)?
                [arg, ii] = getNextArg(options, i)
                # is a number given?
                try:
                    ll = np.float(arg)
                except:
                    ll = None


                if ll is not None:
                    if ll < 0:
                        print('forwardSelect: regularization parameter should be > 0')
                        raise ValueError('forwardSelect: bad value for -g option')
                    else:
                        lamda = ll
                        i = ii

            # ----------------
            elif arg=='-r':
                """ # turn global regularization on (method & number, number, or method only)"""
                Global = True

                # 1. is method specified
                [arg, ii] = getNextArg(options, i)
                method_given = True

                if arg.lower() in ['gcv','uev','fpe','bic']:
                    # use fraction of explained variance, FPE,GCV, BIC to re-estimate
                    ReEst = arg[0].lower()
                else:
                    # no, the method wasn't specified, set default
                    method_given = False
                    ReEst = 'g'

                if method_given:
                    # advance to next arg and read it
                    i = ii
                    [arg, ii] = getNextArg(options, i)


                # 2. is a value given?
                try:
                    ll=np.float(arg)
                except:
                    ll = None

                if ll is not None:
                    # an initial value for lamda is specified
                    if ll >= 0:
                        lamda = ll
                        i = ii
                    else:
                        print('forwardSelect: regularization parameter should be > 0')
                        raise ValueError('forwardSelect: bad value for -r option')

            # ----------------
            elif arg.lower()=='ols':
                """ # turn orthogonal least squares on """
                OLS = True

            else:
                print('%s' %options)
                print((i-len(arg)-1)* [' '] + '^')
                raise ValueError('forwardSelect: unrecognized option')

            # get next argument
            [arg, i] = getNextArg(options, i)



    # initialize
    # ---------------------------
    # set lamda to zero for case of global regularization not used
    if not Global: lamda = 0

    Continue = True
    YY       = traceProduct(Y.conj().T, Y)                                      # [.] trace(Y'Y [n,n])
    AgeOfMin = 0
    m        = 0

    # print out report template
    if Verbose:
        hdr='pass   add    '
        if ReEst != 'n':
            hdr += '  lambda  '
        hdr += '   %s    ' %Model_Dict[Term]
        if Flops:
            hdr +='  flops   '
            flops=0
        print hdr





    # -------------------------------------
    # search for most significant regressors
    #          Main while loop
    # -------------------------------------
    while Continue:
        """ CHECK OLS vs Other Equations """
        # -------------------------------
        # Regressor (First m=1)
        # -------------------------------
        #   H[p,m]    Y[p,n]  F[p,M>>m]    P[p,p]   Y'Y [.]
        if m == 0: # first regressor - initialize
            # (@ Section 7, A.15: Cm-Cm+1 Eq. @)
            Fm       = F                                                        # initially at first regressor Fm --> F0=F[p,M]  (@ 7.1 Pg 32 @)
            ##Um     = 1                                                        # initially at first regressor U1 = 1

            # (@ Section A.15: cost function C @)
            # get error change associated with each regressor
            numerator   = rowSum(np.dot(Fm.conj().T , Y)**2)                    # element-wise power [M,1], orthogonal F'Y = [M,p][p,n] --> Diagonal<>0
            denominator = lamda + diagProduct(Fm.conj().T, Fm)                  # [M,1]
            err         = numerator / denominator                               # [M,1] =[M,1]/[M,1]


            # (@ Section 7.1, Pg 32 @)
            # select the Max change (over 1<J<M) to set the best basis function to be added
            mxerr   = np.max(err)                                               # Scalar [.]
            candid  = np.where(err == mxerr)[0]                                 # Scalar [.]
            choose  = candid[0]                                                 ## candid(1)
            subset  = [choose]                                                  # List
            tot     = mxerr / YY                                                # Vector [M,1]/[.]

            fj      = Fm[:,choose][:, np.newaxis]                               # Vector [p,1]  originial F[:,choose] and not Fm
            fjTfj   = np.dot(fj.conj().T , fj)                                  # Scalar [1,p][p,1]=[.]
            if Verbose: estr = '%4i %5i    ' %(m, choose)


            if OLS:
                # (@ Section 7.1, Pg 32: Orthogonal Least Square @)
                # initialize design matrix
                #   Ho ( with orthogonalized columns) and
                #   Hn (with normalized columns, similar to Ho)
                #   and some other useful stuff
                Hn          = fj / fjTfj                                        # Vector [p,1]              Hn[:,m] =
                Ho          = fj                                                # Vector [p,1]              Ho[:,m] =
                HoTY        = np.dot( fj.conj().T , Y)                          # Vector [1,n]  =[1,p][p,n]
                diagHoTHo   = [[fjTfj]]                                         # Vector [1+,1] =[1,p][p,1]

                Fm = Fm - np.dot(Hn[:,m][:, np.newaxis] , np.dot(Ho[:,m][:, np.newaxis].conj().T , Fm))       # init. Fm ready for second iteration [p,M] = [p,1] [1,p][p,M]  (@ 7.1 Pg 32 @)
                Um = 1                                                          # init. upper triangular matrix Um                              (@ 7.1 Pg 32 @)

            else:
                # (@ Similar??? to Section 7.1, Pg 32: OLS @)
                Hm = fj                                                         # init. Hm = hm [p,m @1]
                Fm = Fm - 1/fjTfj * fj.dot(fj.conj().T).dot(Fm)                 # init. Fm ready for second iteration  (Similar??? to @ 7.1 Pg 32 @)



        # -------------------------------
        # Regressors (Rest after first m>1)
        # -------------------------------
        #   H[p,m]    Y[p,n]  F[p,M>>m]    P[p,p]   Y'Y [.]
        else:
            # (@ Section A.15: cost function C @)
            # get error change associated with each regressor
            numerator   = rowSum(np.dot(Fm.conj().T , Y)**2)                    # element-wise power [M,1]
            denominator = lamda + diagProduct(Fm.conj().T, Fm)                # [M,1]
            denominator[subset] = np.ones((len(subset),1))                      # avoid division by zero
            err         = numerator / denominator                               # [M,1]/[M,1]

            # select the maximum change
            mxerr   = np.max(err)                                               # Scalar [.]
            candid  = np.where(err == mxerr)[0]                                 # Scalar [.]
            choose  = candid[0]                                                 ## candid(1)
            subset.append(choose)                                               # List
            tot     = tot + mxerr / YY                                          # Vector [M,1]/[.]
            fj      = Fm[:,choose][:, np.newaxis]                               # Vector [p,1]
            fjTfj   = np.dot(fj.conj().T , fj)                                  # Scalar [1,p][p,1]=[.]
            if Verbose: estr = '%4i %5i    ' %(m, choose)


            if OLS:
                # (@ Section 7.1 Pg 32 @)
                # collect next column for design matrix
                #   Ho ( with orthogonalized columns) and
                #   Hn (with normalized columns, similar to Ho)
                #   and some other useful stuff
                Hn          = np.hstack( (Hn, fj / fjTfj ) )                    # Vector        [p,1+ m-1]                 Hn[:,m]=
                Ho          = np.hstack( (Ho, fj) )                             # Vector        [p,1+ m-1]                 Ho[:,m]=
                HoTY        = np.vstack( (HoTY, np.dot(fj.conj().T , Y)) )      # Vstack Ho'Y,  [m-1 +1,n]  =[1,p][p,n]
                diagHoTHo   = np.vstack( (diagHoTHo, fjTfj) )                   # List          [m-1 +1,1]  =[1,p][p,1]

                # Fm [p,M]=[p,1][1,p][p,M]
                # Um [m+1,m+1]={[m,m], [m,p][p,1]; [1,m],[1,1]}
                Fm = Fm - np.dot(Hn[:,m][:, np.newaxis] , np.dot(Ho[:,m][:, np.newaxis].conj().T , Fm))     # recompute Fm ready for next iteration (@ Section 7.1, Pg 32)
                Um = np.bmat([ [Um, np.dot(Hn[:,0:m].conj().T , F[:,choose][:, np.newaxis])], \
                                [np.zeros((1,m)), 1] ])                         # update Um = [ [Um_1, Hn[:,0:m]'Fj] ; [0_m-1 1] ], Hn= Hm/h'h      (@ Section 7.1, Pg 32)
            else:
                # (@ Similar??? to Section 7.1 Pg 32: OLS @)
                Hm = np.hstack( (Hm, F[:,choose][:, np.newaxis]) )              # update H  = [H_m-1, hm], hm=[p,1]
                Fm = Fm - 1/fjTfj * fj.dot(fj.conj().T).dot(Fm)                 # recompute Fm ready for next iteration (Similar??? to @ Section 7.1, Pg 32 )


        # -------------------------------
        # Re-estimate lamda (with fj)
        #           -r option
        # -------------------------------
        #   H[p,m]    Y[p,n]  A[m,m]    W[m,n]    F[p,M>>m]    P[p,p]   Y'Y [.]
        if ReEst != 'n':
            # calculate Gamma (g) needed for Lambda re-estimation
            if OLS:
                # (@ Section A.15: Regularized OLS @)
                YTHoTHoY    = diagProduct(HoTY,HoTY.conj().T)                   # Y'Ho'HoY = Diag(Ho'Y,(Ho'Y)')     [m,1]=Diag([m,n][n,m])
                ldiagHoTHo  = lamda + diagHoTHo                                 # OK, used below Diag(Ho'Ho) [m,1]=Diag([m,p][p,m])
                ldiagHoTHo2 = ldiagHoTHo * ldiagHoTHo                           # OK, used below   element prod     [m,1]
                ldiagHoTHo3 = ldiagHoTHo2 * ldiagHoTHo                          # OK, used below   element prod     [m,1]
                l2diagHoTHo = lamda + ldiagHoTHo                                # OK, used below                    [.]
                YP2Y        = YY - np.sum(l2diagHoTHo * YTHoTHoY / ldiagHoTHo2) # Sm - Sm+1 = Y'(Pm^2-Pm+1^2)Y      [.]         (@ Section 7.3 Eq 7.9, A.15, Pg 64)
                WAW         = np.sum(YTHoTHoY / ldiagHoTHo3)                    # W = Y'Ho/(l+Ho'Ho),    sum([m,1])=[.]         (@ Section A.15, Pg 64)
                trA         = np.sum(1 / ldiagHoTHo)                            # A= Ho'Ho+l= Diag[1/(l+Ho'Ho)]     [.]         (@ Section A.15, Pg 64)
                trA2        = np.sum(1 / ldiagHoTHo2)                           # A= Ho'Ho+l= Diag[1/(l+Ho'Ho)]     [.]         (@ Section A.15, Pg 64)
                g           = np.sum(diagHoTHo / ldiagHoTHo)                    # g= Ho'Ho/(l+Ho'Ho) sum([m,1]/[m,1])=[.]       (@ Section A.15, Pg 65)

            else:
                # {PDF A^-1 -->Code A}
                A       = np.linalg.inv( np.dot(Hm.conj().T, Hm) + lamda*np.eye(m))   # A^-1=(H'H+LAMBDA)^-1 [m,m]=[m,p][p,m] (@ Section 4.1: Eq 4.4)
                HY      = np.dot(Hm.conj().T , Y)                               # OK [m,n]=[m,p][p,n]
                W       = np.dot(A , HY)                                        # W = A^-1 H'Y              [m,n]=[m,m][m,n]    (@ Section 4.1: Eq 4.5)
                PY      = Y - np.dot(Hm , W)                                    # PY= Y-F= Y-HW Error Vect  [p,n]-[p,m][m,n]    (@ Section A.6: Pg 45)
                YP2Y    = traceProduct(PY.conj().T, PY)                         # OK                        [.]
                WAW     = traceProduct( W.conj().T, np.dot(A , W) )             # OK                        [.]
                trA     = np.trace(A)                                           # OK                        [.]
                trA2    = np.trace(np.dot(A,A))                                 # OK                        [.]
                g       = m - lamda * trA                                       # g[.] = p-tr(P)= tr(A^-1H'H)= m-l*tr(A^-1) (@ Section A.8, Eq 4.10)


            # (@ Section A.10, Pg 57 @)
            # exercise different re-estimation methods
            if ReEst == 'g':
                # GCV method
                psi = 1 / (p-g)

            elif ReEst == 'u':
                # UEV method
                psi = 1 / (2*(p-g))

            elif ReEst == 'f':
                # FPE method
                psi = p / ((p-g)*(p+g))

            else:
                # BIC method
                psi = p*np.log(p) / (2*(p-g)* (p + (np.log(p)-1) * g))

            # do the re-estimation of Lambda
            lamda = psi * YP2Y * (trA - lamda * trA2) / WAW                     # [.] = [.][.]*[.]/[.]
            if Verbose:  estr += '%8.3e ' %lamda

            # Append re-estimated l to keep track of multiple lamdas
            if m == 0:
                lamdas = [lamda]
            else:
                lamdas = lambddas.append(lamda)                                 # List [.]




        # -------------------------------
        # calculate current score (with fj)
        #           -t option
        # -------------------------------
        #   H[p,m]    Y[p,n]  A[m,m]    W[m,n]    F[p,M>>m]    P[p,p]   Y'Y [.]
        eps = 1e-12         # eps = np.spacing(1)

        # LOO Termination
        # -------------------
        if Term == 'l':
            # (@ Section A.9 LOO @)
            if OLS:
                # calculate diag( Pm = Ip - HAH')
                ldiagHoTHo  = lamda + diagHoTHo                                             # OK, used below Diag(Ho'Ho) [m,1]=Diag([m,p][p,m])
                diagPm      = np.ones((p,1)) - \
                            diagProduct(Ho, dupCol(1/ldiagHoTHo,p) * Ho.conj().T )          # A = Ho'Ho+l = Diag[1/(l+Ho'Ho)]           (@ Section A.15, Pg 64)
                                                                                            # [p,1]=[p,1]-Diag([p,m] [m,p].*[m,p])
                # watch out for zero entries along the diagonal of P
                tooSmall        = np.where(diagPm < eps)[0]
                diagPm[tooSmall]= eps * np.ones((len(tooSmall),1))

                # (@ Section A.6: Pg 45 @)
                # need Pm*Y & inv(diag( diag(Pm))) * Pm * Y
                PmY = Y - np.dot(Ho, (np.dot(Ho.conj().T, Y))/dupCol(ldiagHoTHo,n) )        # PY= Y-HoHo'Y/(l+Diag(Ho'Ho)) Error [p,n]=[p,m][m,p][p,n]
                invDiagPmPmY = PmY / dupCol(diagPm, n)                                      #                                    [p,n]

                # compute LOO (Sherman-Morrison-Woodbury formula)
                score = traceProduct(invDiagPmPmY.conj().T, invDiagPmPmY) / p               # sigma2_LOO         [.]=Trace([n,p][p,n])  (@ Section A.9 Pg 54)

            else:
                A       = np.linalg.inv( np.dot(Hm.conj().T, Hm) + lamda * np.eye(m))       # A^-1=(H'H+LAMBDA)^-1 [m,m]=[m,p][p,m]     (@ Section 4.1: Eq 4.4)
                HY      = np.dot(Hm.conj().T , Y)                                           # Ok                 [m,n]=[m,p][p,n]
                AH      = np.dot(A , Hm.conj().T)                                           # OK                 [m,p]=[m,m][m,p]
                PY      = Y - Hm.dot(AH).dot(Y)                                             # PY=Y-HW Err Vect   [p,n]=[p,m][m,n][p,n]  (@ Section A.6: Pg 45)
                dP      = np.ones((p,1)) - diagProduct(Hm, AH)                              # dP= 1 - Diag(HAH') [p,1]=diag([p,m][m,p]) (@ Section 4.1: Eq 4.6)
                dPPY    = PY / dupCol(dP, n)                                                # OK used below      [p,n]=[p,n]/[p,1*n]
                score   = traceProduct(dPPY.conj().T, dPPY) / p                             # sigma2_LOO         [.]=Trace([n,p][p,n])  (@ Section A.9 Pg 54)

        # Non-LOO Termination
        # -------------------
        else:
            # (@ Section 4.4, A.8 @)
            # get trace(Pm) and also AH if not using OLS
            if lamda == 0:
                if OLS:
                    tracePm = p - m                                                         # Scalar [.]
                else:
                    # lamda = 0
                    A  = np.linalg.inv( np.dot(Hm.conj().T , Hm))                           # A^-1=(H'H)^-1      [m,m]=[m,p][p,m]       (@ Section 4.1: Eq 4.4)
                    AH = np.dot( A , Hm.conj().T)                                           # OK                 [m,p]=[m,m][m,p]
                    tracePm = p - m                                                         # Scalar [.]
            else:
                if OLS:
                    tracePm = p - np.sum(diagHoTHo / (lamda + diagHoTHo))                   # Scalar [.] = sum(Diag([m,p][p,m])
                else:
                    A  = np.linalg.inv( np.dot(Hm.conj().T , Hm)+ lamda * np.eye(m))        # A^-1=(H'H)^-1      [m,m]=[m,p][p,m]       (@ Section 4.1: Eq 4.4)
                    AH = np.dot( A , Hm.conj().T)                                           # OK                 [m,p]=[m,m][m,p]
                    tracePm = p - traceProduct(Hm, AH)                                      # Scalar [.]


            # Get number of effective parameters (g)=p-Trace(P)
            # Watch out for zero trace (usually p = m and lambda = 0).
            if tracePm < eps:
                tracePm = eps
                g = p                                                                       # Scalar [.]
            else:
                g = p - tracePm                                                             # Scalar [.]


            # get mean square error
            if OLS:
                if lamda == 0:
                    YP2Yp       = (YY - np.sum(diagProduct(HoTY,HoTY.conj().T) / diagHoTHo)) / p        # Scalar [.]-sum( Diag([m,n][n,m])/Diag([m,p][p,m]))
                    YP2Yp       = (YY - traceProduct(HoTY.conj().T, HoTY / dupCol(diagHoTHo, n))) / p
                else:
                    ldiagHoTHo  = lamda + diagHoTHo                                                     # OK, used below Diag(Ho'Ho) [m,1]=Diag([m,p][p,m])
                    ldiagHoTHo2 = ldiagHoTHo**2                                                         # OK, used below   element prod     [m,1]
                    YP2Yp       = (YY - traceProduct(HoTY.conj().T, HoTY / dupCol(ldiagHoTHo, n))    \
                    - lamda * traceProduct(HoTY.conj().T, HoTY / dupCol(ldiagHoTHo2, n))) / p           # OK Scalar        [.]=Trace([n,p][p,n])

            else:
                PY      = Y - np.dot(Hm , np.dot(AH , Y) )                                              # PY=Y-HW=Y-HAHY [p,n]=[p,m][m,n][p,n]  (@ Section A.6: Pg 45)
                YP2Yp   = traceProduct(PY.conj().T, PY) / p                                             # OK Scalar        [.]=Trace([n,p][p,n])

            # get different factors for each method
            if Term == 'v':
                # unexplained variance
                psi = 1

            elif Term == 'g':
                # GCV
                psi = p**2 / tracePm**2

            elif Term == 'u':
                # UEV
                psi = p / tracePm

            elif Term == 'f':
                # FPE
                psi = (p + g) / tracePm

            else:
                # BIC
                psi = (p + (np.log(p) - 1) * g) / tracePm

            # finally compute score
            score = psi * YP2Yp                                                 # Sigma2 Scalar [.]

        if Verbose: estr += '%9.3e ' %score


        # -------------------------------
        # Wrap-up While
        # -------------------------------

          # are we ready to terminate yet
        if tot >= 1:
            #
            Continue = False
            if Flops:   estr += '%8i  ' %flops
            if Verbose: estr += '\n ==> variance all explained '

        elif m >= M:
            # all columns of F[p,M>>m] tried
            Continue = False
            if Flops:   estr += '%8i  ' %flops
            if Verbose: estr += '\n ==> regressors used up '

        elif MaxReg > 0 and  m >= MaxReg:
            # m > Max Reg
            Continue = False
            if Flops:   estr += '%8i  ' %flops
            if Verbose: estr += '\n ==> limit of regressors reached '

        else:
            # decide if termination conditions have been met
            if m == 0:
                # don't stop here unless threshold being used
                if Term == 'v':
                    if (1 - p * score / YY) > Threshold:
                        Continue = False
                        if Flops:   estr += '%8i  ' %flops
                        if Verbose: estr += 'explained variance threshold exceeded '
                else:
                    MinScore = score
                    AgeOfMin = 1

            else:
                # decide between threshold and minimum methods
                if Term == 'v':
                    if (1 - p * score / YY) > Threshold:
                        Continue = False
                        if Flops:   estr += '%8i  ' %flops
                        if Verbose: estr += '\n ==> explained variance threshold exceeded '

                else:
                    # compare old and new score and age of minimum
                    if score < MinScore:
                        # new minimum - don't stop here
                        MinScore = score
                        AgeOfMin = 1

                    else:
                        if AgeOfMin >= MaxAge:
                            # old minimum has gone on long enough - stop here
                            Continue = False
                            if Flops:   estr += '%8i  ' %flops
                            if Verbose: estr += '\n ==> minimum passed '
                        else:
                            # old minimum just ages by 1
                            AgeOfMin += 1

        if Flops and Continue:  estr += '%8i  ' %flops
        if Verbose:             print estr
        m += 1          # increment number of regressors
        # -------------------------------
        # end while
        # -------------------------------



    # -------------------------------
    # Prepare Outputs (Post-While)
    # -------------------------------
    #   H[p,m]    Y[p,n]  A[m,m]    W[m,n]    F[p,M>>m]    P[p,p]   Y'Y [.]
    # don't include last few regressors which aged the minimum
    m       = m - AgeOfMin
    subset  = subset[0:m]                                                       ## subset(1:m)

    # actual design matrix being used
    H = F[:, subset]                                                            # H[p,m]    subset=range()

    # regularization constant
    if ReEst == 'n':
        l = lamda                                                               # l[.]
    else:
        l = lamdas[m]


    if OLS:
        # truncate OLS structures (in case minimum aged and m shrank)
        Um        = Um[0:m, 0:m]                                                # U[m,m]                                ## Um(1:m, 1:m)
        diagHoTHo = diagHoTHo[0:m,:]                                            # Diag(Ho'Ho) [m,1]=Diag([m,p][p,m])    ## diagHoTHo(1:m)
        HoTY      = HoTY[0:m,:]                                                 ## HoTY(1:m,:)

        # upper triangular matrix
        U = Um

        # covariance
        invU = np.linalg.inv(U)                                                 # OK used below
        Ao   = np.diag(1 / (diagHoTHo + lamda))                                 # (@ Section 7.3 Pg 34,  A.15 Pg 64)
        A    = invU.dot(Ao).dot(invU.conj().T)                                  # (@ Section 7.3, Pg 33)

        # (@ Section 7.1, A.13, A.15 @)
        # weight vector (or matrix)
        wo = np.dot(Ao , HoTY)                                                  # w = AHo'Y (@ Section 4.1, Eq 4.5)
        w  = np.dot(invU , wo)                                                  # (@ Section A.15, Pg 64)

        # projection matrix
        # P = Ip - HAH'
        # A = Ho'Ho+l = Diag[1/(l+Ho'Ho)]                                       (@ Section 7.3 Pg 34,  A.15 Pg 64)
        P = np.eye(p)                                                           # P=I[p,p]
        for j in range(m):                                                      # P[p,p]=[p,1][1,p]/[.]     (@ Section A.15, Pg 64)
            P = P - np.dot( Ho[:,j][:, np.newaxis] , Ho[:,j][:, np.newaxis].conj().T) / (lamda + diagHoTHo[j])

    else:
        U = np.eye(m)                                                           # OK  defined as Im [m,m]
        A = np.linalg.inv( np.dot(H.conj().T, H) + lamda * np.eye(m))           # A = (H'H+l*I)^-1  [m,m]=[m,p][p,m]        (@ Section 4.1: Eq 4.4)
        w = A.dot(H.conj().T).dot(Y)                                            # w = AH'Y          [m,n]=[m,m][m,p][p,n]   (@ Section 4.1 Eq 4.5 )
        P = np.eye(p) - H.dot(A).dot(H.conj().T)                                # P = I - HAH'      [p,p]=[p,m][m,m][m,p]   (@ Section 4.1 Eq 4.6 )


    return [subset, H, l, U, A, w, P]

# ==============================================================================
#                              MAIN RUN PART
# ==============================================================================
if __name__ == '__main__':
    pass
