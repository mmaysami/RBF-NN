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



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#===============================================================================
#           Predict Error
#===============================================================================
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def predictError(H, Y, l, options=None, U=None):
    """
    # [e, W, A, P, g] = predictError(H, Y, l, options, U)
    #
    # PDF: Sections 5 - Model Selection Critria (& 4.4)
    # Calculates the predicted error on a future test set for a linear
    # network of design H with output training points Y and using either
    # local or global ridge regression with regularisation parameter(s) l (Lambda).
    # Uses a number of alternative methods:
    #
    #   options = MSE: Mean Square Error of training set    (m)
    #   options = GCV: Generalized Cross-Validation         (g)
    #   options = UEV: Unbiased Estimate of Variance        (u)
    #   options = FPE: Final Prediction Error               (f)
    #   options = BIC: Bayesian Information Criterion       (b)
    #   options = LOO: Leave-One-Out cross-validation       (l)
    #
    # The options string can contain one or more of above substrings
    # (seperated by commas or spaces) in which case the result will
    # be a list (row vector) of the corresponding error estimates.
    #
    #   n: Size of Inputs               [x1 .. xn]
    #   m: Size of Bases                [h1 .. hm]
    #   p: Number of Training Samples   {X,y}p
    #   k: Number of Test     Samples   {X,y}k
    #
    # Inputs
    #
    #   H       design matrix                       (p-by-m)
    #   Y       input trainig data                  (p-by-k)
    #   l       Lambda regularisation parameter(s)  (real or vector length m)
    #   options error prediction method(s)          (string)
    #   U       alternative smoothing metric        (m-by-m)
    #           UU = U'*L*U
    #
    # Outputs
    #
    #   e       predicted error                     (row vector)
    #   W       A * H' * Y                          (m-by-k)
    #   A       inv(H' * H + L)                     (m-by-m)        {PDF A^-1 -->Code A}
    #   P       I - H  * A * H'                     (p-by-p)
    #   g       gamma = p - trace(P)                (real)
    #
    """

    # no model to begin with
    Model = ''

    # process options
    # ---------------------------
    if options is not None:
        # initialize
        i = 0                                                                   ## i =1
        [arg, i] = getNextArg(options, i)

        # scan through arguments
        while arg != '':
            if arg.lower() in ['mse','gcv''uev','fpe','bic','loo']:
                # set Variance error criteria
                Model += arg[0].lower()
            else:
                print('predictError: use MSE, UEV, FPE, GCV, BIC or LOO')
                raise AssertionError('predictError: bad option')

            # get next argument
            [arg, i] = getNextArg(options, i)


    # default model if None specified in options
    if Model=='':
        Model = 'g'


    if U is None:
        U = np.eye(1)         # default metric will be eye(m)


    # initialize L & UU = U'*L*U
    # ---------------------------
    #   n: Size of Inputs               [x1 .. xn]
    #   m: Size of Bases                [h1 .. hm]
    #   p: Number of Training Samples   {X,y}p
    #   k: Number of Test     Samples   {X,y}k
    #
    #   H[p,m]    Y[p,k]  l[1,m]  U[m,m]

    U = np.atleast_2d(U)    # atleast_2d makes 1D to [1,n]

    [p, m]   = H.shape
    [p, k]   = Y.shape
    [u1, u2] = U.shape


    # construct Lambda matrix
    if not isinstance(l, (np.ndarray, list)):
        L = np.diagflat(l * np.ones((m,1)))
    elif len(l) == 1:
        L = np.diagflat(l * np.ones((m,1)))
    elif len(l) == m:
        L = np.diagflat(l)
    else:
        raise AssertionError('predictError: wrongly sized regularisation parameter')

    # construct regularization matrix UU = U'*L*U
    if (u1==1) and (u2==1):
        UU = L
    elif (u1==m) and (u2==m):
        UU = U.conj().T.dot(L).dot(U)
    else:
        raise AssertionError(['predictError: U should be 1-by-1 or %s-by-%s' %(m,m)])



    # preliminary calculations
    # ---------------------------
    #   H[p,m]      Y[p,k]      L[m,m]      U[m,m]
    HH  = np.dot(H.conj().T, H)                         # HH [m,m] = H'*H
    HY  = np.dot(H.conj().T, Y)                         # HY [m,k] = H'*Y

    A   = np.linalg.inv(HH + UU)                        # A  [m,m] = (H'H + U'LU)^-1    {PDF A^-1 -->Code A}
    W   = np.dot(A, HY)                                 # W  [m,k] = A * H'*Y = A * HY
    P   = np.eye(p) - H.dot(A).dot(H.conj().T)          # P  [p,p] = Ip - HAH'

    PY  = np.dot(P, Y)                                  # PY [p,k] = P*Y                ~ y^ - f
    YPY = traceProduct(PY.conj().T, PY)                 # YPY[.]   = trace(PY' * PY)    ~ S^
    g   = p - np.trace(P)                               # g  [.]   = p - trace(P)


    # -------------------------------------
    # calculate errors for each method
    #       specified in options
    # -------------------------------------
    # (@ Section 5.2: Eq. 5.1-5.6 @)
    e = []
    for model in Model:
        # Leave-One-Out (LOO) cases
        if model == 'l':
            # special case of LOO
            dPPY = PY / dupCol(np.diag(P), k)
            em   = traceProduct(dPPY.conj().T, dPPY) / p

        # All other model cases
        #  e = psi * S^ / p = psi * Y'P^2Y / p
        else:
            # value of factor psi
            #   p-g = trace(P)
            if model == 'm':
                # MSE
                psi = 1

            elif model == 'g':
                # GCV
                psi = p**2 / (p - g)**2

            elif model == 'u':
                # UEV
                psi = p / (p - g)

            elif model == 'f':
                # FPE
                psi = (p + g) / (p - g)
            else:
                # BIC
                psi = (p + (np.log(p)-1) * g) / (p - g)

            em = psi * YPY / p

        # final calculation
        e.append(em)


    # {PDF A^-1 -->Code A}
    return [e, W, A, P, g]



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#===============================================================================
#           Global Ridge
#===============================================================================
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def globalRidge(H, Y, l= 0.01, options=None, U=None):
    """
    # [l, e, L, E] = globalRidge(H, Y, l, options, U)
    #
    # @ PDF Section A.10 - A ReEstimation Formula for the Global Parameter
    # Calculates the best global ridge regression parameter (l) and
    # the corresponding predicted error (e) using one of a number of
    # prediction methods (UEV, FPE, GCV or BIC). Needs a design (H),
    # the training set outputs (Y), and an initial guess (l).
    #
    # The termination criterion, maximum number of iterations,
    # verbose output and the use of a non-standard weight penalty
    # are controlled from the options string. The non-standard
    # metric, if used, is given in the fifth argument (U). L and E
    # return the evolution of the regularization parameter and error
    # values from the initial to final iterations.
    #
    # If the input l is a vector (more than one guess), a
    # corresponding number of answers will be returned, e will
    # also be a vector and L and E will be matrices (with each row
    # corresponding to the iterations resulting after each guess).
    #
    #   n: Size of Inputs               [x1 .. xn]
    #   m: Size of Bases                [h1 .. hm]
    #   p: Number of Training Samples   {X,y}p
    #   k: Number of Test     Samples   {X,y}k
    #
    # Inputs
    #
    #   H        design matrix                          (p-by-m)
    #   Y        input training data                    (p-by-k)
    #   l        initial guess(es) at lambda [def:0.01] (vector length q)
    #   options  options                                (string)
    #           -v          Verbose = True
    #           -V          Verbose, Flops = True
    #           -h #>1      Hard limit of iters = #
    #           -t #>0      Threshold value
    #           uev/gcv/... Variance error criteria
    #
    #   U        optional non-standard smoothing metric (m-by-m)
    #
    # Outputs
    #
    #   l        final estimate(s) for lambda                (1-by-q)
    #   e        final estimate(s) for model selection score (1-by-q)
    #   L        list(s) of running lambda values            (n-by-q)
    #   E        list(s) of running error values             (n-by-q)
    #
    # The various model selection criteria used are:
    #
    #   GCV  Generalized Cross Validation
    #   UEV  Unbiased Estimate of Variance
    #   FPE  Final Prediction Error
    #   BIC  Bayesian Information Criterion
    #
    # specified in options by, e.g. 'FPE'.
    """

    # defaults
    # ---------------------------
    Model_Dict  = {
                    'g':'GCV',
                    'u':'UEV',
                    'f':'FPE',
                    'b':'BIC',
                    'l':'LOO',
                  }

    Model       = 'g'   # GCV Error Model
    Verbose     = False # Verbosity
    Flops       = False # Compute Cost
    Standard    = True  # No U
    Hard        = 100
    Threshold   = 1000
    # q         : number of initial guesses


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

            elif arg=='-U':
                # non-standard penalty matrix
                Standard = False

            elif arg=='-h':
                """ # hard limit to specify """
                [arg, i] = getNextArg(options, i)
                try:
                    hl = float(arg)
                except:
                    hl = None

                if hl is not None:
                    if hl > 1:
                        Hard = np.round(hl)
                    else:
                        print('globalRidge: hard limit should be positive > 1')
                        raise ValueError('globalRidge: bad value in -h option')
                else:
                    print('globalRidge: value needed for hard limit')
                    raise ValueError('globalRidge: missing value in -h option')

            elif arg=='-t':
                """ # termination criterion to specify """
                [arg, i] = getNextArg(options, i)
                try:
                    te = np.float(arg)
                except:
                    te = None

                if te is not None:
                    if te >= 1:
                        Threshold = np.round(te)
                    elif te > 0:
                        Threshold = te
                    else:
                        print('globalRidge: threshold should be positive')
                        raise ValueError('globalRidge: bad value in -t option')
                else:
                    print('globalRidge: value needed for threshold')
                    raise ValueError('globalRidge: missing value in -t option')


            elif arg.lower() in ['gcv','uev','fpe','bic']:
                """ # set Variance error criteria """
                Model = arg[0].lower()

            else:
                print('%s' %options)
                print((i-len(arg)-1)* [' '] + '^')
                raise ValueError('globalRidge: unrecognized option')

            # get next argument
            [arg, i] = getNextArg(options, i)


    # process other optional arguments
    # ---------------------------
    # Initialize Lambda
    if l is None:
        l = 0.01 # default initial guess

    # Initialize U (Non-Standard Smoothing)
    if not Standard:
        if U is None:
            print('globalRidge: specify non-standard penalty matrix')
            raise ValueError('globalRidge: -U option implies fifth argument')
    else:
        U = np.eye(1)



    # initialize
    # ---------------------------
    #   n: Size of Inputs               [x1 .. xn]
    #   m: Size of Bases                [h1 .. hm]
    #   p: Number of Training Samples   {X,y}p
    #   k: Number of Test     Samples   {X,y}k
    #
    #   H[p,m]      Y[p,k]      l[q]      U[m,m]

    l = np.atleast_2d(l)    # atleast_2d makes 1D to [1,n]
    U = np.atleast_2d(U)    # atleast_2d makes 1D to [1,n]

    [p, m]   = H.shape
    [p, k]   = Y.shape
    [q1, q2] = l.shape
    [u1, u2] = U.shape

    if q1>1 and q2>1:
      raise AssertionError('globalRidge: list of guesses should be vector, not matrix')
    else:
        q= np.max([q1,q2])
    # for Python l = [q,.]
    l = np.ravel(l)

    if (u1==m) and (u2==m):
        # transform the problem - equivalent to U'*U metric
        # u1=u2=1 will ignore U
        H = H * np.linalg.inv(U)
    elif (u1!=1) or (u2!=1):
        raise AssertionError('globalRidge: U should be 1-by-1 or %s-by-%s' %(m,m))



    # preliminary calculations
    # ---------------------------
    #   H[p,m]      Y[p,k]      l[q]      U[m,m]
    HH  = np.dot(H.conj().T, H)                         # HH [m,m] = H'*H
    HY  = np.dot(H.conj().T, Y)                         # HY [m,k] = H'*Y
    e   = np.zeros((1, q))
    # for Python e = [q,.]
    e = np.ravel(e)

    ##if nargout > 2 :
    L = np.zeros((Hard+1, q))   # Hard:iter/guess, q: #initial guesses
    ##if nargout > 3
    E = np.zeros((Hard+1, q))   # Hard:iter/guess, q: #initial guesses
    maxcount = 1

    if Verbose:
      print('%s Global Ridge %s' %('\n','\n'))

    if Flops:
        flops= 0
        ##flops(0)


    # ------------------------------------------------------
    # loop through each initial guess
    # ------------------------------------------------------
    for i in range(q):
        # print out report template
        if Verbose:
            hdr='pass '+'  lambda  '+'   %s    ' %Model_Dict[Model]+' change '
            if Flops: hdr +='  flops'
            print hdr

        # reset counter
        count   = 0

        # (Sections 4.1- 4.4: Eq. 4.4 - 4.10)
        # calculate model selection score
        #   H[p,m]      Y[p,k]      l[vector q]
        A   = np.linalg.inv(HH + l[i] * np.eye(m))  # A  [m,m] = (H'H + li * Im)^-1     {PDF A^-1 -->Code A}
        g   = m - l[i] * np.trace(A)                # g  [.]   = m - l[.] * trace(A)    (@ Section A.8)
        PY  = Y - H.dot(A).dot(HY)                  # PY [p,k] = (Ip - HAH') * Y
        YPY = traceProduct(PY.conj().T, PY)         # YPY[.]   = trace(PY' * PY)        ~ S^


        # (@ Section 5.2: Eq. 5.1-5.6 @)
        # set error calculation based on model
        if Model == 'g':
            psi = p**2 / (p - g)**2
        elif Model == 'u':
            psi = p / (p - g)
        elif Model == 'f':
            psi = (p + g) / (p - g)
        else:
            psi = (p + (np.log(p)-1) * g) / (p - g)

        e[i] = psi * YPY / p

        if Verbose:
            estr = '%4i %9.3e %9.3e       - ' %(count, l[i], e[i])
            if Flops: estr += '%9i' #flops
            print estr

        if l is not None:       L[1,i] = l[i]
        if options is not None: E[1,i] = e[i]

        # reset action flags
        TooMany = False
        Done    = False

        # re-estimate til convergence or exhaustion of iterations
        # ------------------------------------------------------
        while not Done and not TooMany:
            # next iteration
            count += 1

            # get some needed quantities
            # {PDF A^-1 -->Code A}
            A2 = np.dot(A,A)
            A3 = np.dot(A,A2)

            # (@ Section A.10 @)
            # re-estimate new lambda
            if Model == 'g':
                eta = 1 / (p - g)
            elif Model == 'u':
                eta = 1 / (2 * (p - g))
            elif Model == 'f':
                eta = p / ((p - g) * (p + g))
            else:
                eta = p * np.log(p) / (2 * (p - g) * (p + (np.log(p)-1) * g))

            # new lambda {PDF A^-1 -->Code A}
            nl = eta * YPY * np.trace(A - l[i] * A2) / np.trace(HY.conj().T.dot(A3).dot(HY))

            # store result
            if l is not None:       L[count+1,i] = nl


            # REDO new Lambda (Sections 4.1- 4.4: Eq. 4.4 - 4.10)
            # calculate new model selection score (with new lamba: nl)
            #   H[p,m]      Y[p,k]      l[vector q]
            A   = np.linalg.inv(HH + nl * np.eye(m))    # A  [m,m] = (H'H + li * Im)^-1  {PDF A^-1 -->Code A}
            g   = m - nl * np.trace(A)                  # g  [.]   = m - l[.] * trace(A)
            PY  = Y - H.dot(A).dot(HY)                  # PY [p,k] = (Ip - HAH') * Y
            YPY = traceProduct(PY.conj().T, PY)         # YPY[.]   = trace(PY' * PY)    ~ S^


            # REDO new Lambda (@ Section 5.2: Eq. 5.1-5.6 @)
            # calculate psi factor by model
            if Model == 'g':
                psi = p**2 / (p - g)**2
            elif Model == 'u':
                psi = p / (p - g)
            elif Model == 'f':
                psi = (p + g) / (p - g)
            else:
                psi = (p + (np.log(p) - 1) * g) / (p - g)
            # new score
            ns = psi * YPY / p

            # store result
            if options is not None: E[count+1,i] = ns


            # what's the change (time to go home?)
            if count >= Hard:
                TooMany = True
            elif Threshold >= 1:
                # interpret threshold as one part in many
                change = np.round(np.abs(e[i] / (e[i] - ns)))
                if change > Threshold:
            	   Done = True
            else:
                # interpret threshold as absolute difference
                change = np.abs(e[i] - ns)
                if change < Threshold:
            	   Done = True

            # get ready for next iteration (or end)
            l[i] = nl
            e[i] = ns

            if Verbose:
                estr ='%4i %9.3e %9.3e ' %(count, l[i], e[i])
                if Threshold >=1:
            	    estr += '%7i ' %change
                else:
                    estr += '%7.1e ' %change
                if Flops:
                    estr += '%9i ' %flops
                print estr


        # outside while iterations for each guess
        # ------------------------------------------------------
        if Verbose:
            if  TooMany:
                print('hard limit reached')
            else:
                if Threshold >=1:
                    estr = 'relative'
                else:
                    estr = 'absolute'

                estr += ' threshold in '+ Model_Dict[Model] + ' crossed'
                print estr


        if count > maxcount:
            maxcount = count



    # Outside Loop (Wrap-Up)
    # ------------------------------------------------------
    # truncate L and S
    ##if nargout > 2:
    L = L[0:maxcount+1,:]       #[1:maxcount+1,:]
    ##if nargout > 3:
    E = E[0:maxcount+1,:]       #[1:maxcount+1,:]

    return [l, e, L, E]



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#===============================================================================
#           Local Ridge
#===============================================================================
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def localRidge(F, Y, l=0.01, options=None):
    """
    # [l, V, H, A, W, P] = localRidge(F, Y, l, options)
    #
    # @ PDF Section 6.3 - Local Ridge Regression
    # Calculates the best local ridge regression parameters using
    # one of a number of model selection criteria. Uses an initial
    # guess (l), a termination condition (t) and a hard limit to
    # the number of iterations (s) (all of which have defaults).
    #
    #   n: Size of Inputs               [x1 .. xn]
    #   m: Size of Bases                [h1 .. hm]
    #   M: Size of all Candidate basis  [f1 .. fM], M>>m
    #   p: Number of Training Samples   {X,y}p
    #
    # Inputs
    #
    #   F        full design matrix                             (p-by-M)
    #   Y        input trainig data                             (p-by-n)
    #   l        initial guess at lambdas [default all 0.01]    (1-by-M or scalar)
    #   options  options (string)
    #           -v          Verbose = True
    #           -V          Verbose, Flops = True
    #           -e          Random order =True
    #           -h #>1      Hard limit of iters = #
    #           -t uev/gcv  Terminate method criteria
    #           -t #>0      Threshold value
    #
    #
    # Outputs
    #
    #   l        final estimate for lambdas                 (1-by-M)    {with m finite entries}
    #   V        final estimate for model selection score   (scalar)
    #   H        final design matrix                        (p-by-m)
    #   A        final partial covariance                   (m-by-m)    {PDF A^-1 -->Code A}
    #   W        final partial weight matrix                (m-by-n)
    #   P        final projection matrix                    (p-by-p)
    #
    # The two model selection criteria that can be used are:
    #     GCV    Generalized Cross-Validation
    #     UEV    Unbiased Estimate of Variance

    """

    # defaults
    # ---------------------------
    Model_Dict  = {
                    'g':'GCV',
                    'u':'UEV',
                    'f':'FPE',
                    'b':'BIC',
                    'l':'LOO',
                  }

    Term        = 'g'   # GCV Error Model
    Verbose     = False # Verbosity
    Flops       = False # Compute Cost
    Random      = False
    Hard        = 100
    Threshold   = 1000


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

            elif arg=='-r':
              # random ordering
              Random = True

            elif arg=='-h':
                # hard limit to specify
                [arg, i] = getNextArg(options, i)
                try:
                    hl = np.float(arg)
                except:
                    hl = None

                if hl is not None:
                    if hl > 1:
                        Hard = np.round(hl)
                    else:
                        print('localRidge: hard limit should be positive')
                        raise ValueError('localRidge: bad value in -h option')
                else:
                    print('localRidge: value needed for hard limit')
                    raise ValueError('localRidge: missing value in -h option')


            elif arg=='-t':
                # specify termination criterion (method or number)
                # -----------------------------
                [arg, ii] = getNextArg(options, i)

                # 1. Check for method first
                if arg.lower() in ['gcv','uev','fpe','bic']:
                    #  use UEV, FPE, GCV, BIC to terminate
                    Term = arg[0].lower()
                    method_given = True
                else:
                    # the method wasn't specified, or specified incorrectly
                    method_given = False


                # is a method given?
                if method_given:
                    # skip to next argument
                    i = ii
                    [arg, ii] = getNextArg(options, i)

                # 2. is a number given?
                try:
                    nu = np.float(arg)
                except:
                    nu = None

                if nu is not None:
                    value_given  = True
                    good_value   = True

                    # a numeric value has been specified
                    if (nu>=1):
                        Threshold = np.round(nu)
                    elif (nu>0):
                        Threshold = nu
                    else:
                        good_value = False
                else:
                    value_given = False

                if value_given:
                    i = ii

                # error conditions
                if  not method_given and not value_given:
                    print('localRidge: terminate with UEV or GCV and provide a threshold')
                    error('localRidge: missing arguments for -t option')
                elif value_given and not good_value:
                    print('localRidge: acceptable thresholds are')
                    print('  between 0 and 1 (absolute change)')
                    print('  greater than  1 (relative change)')
                    raise ValueError('localRidge: bad value for -t option')

            else:
                print('%s' %options)
                print((i-len(arg)-1)* [' '] + '^')
                raise ValueError('localRidge: unrecognized option')

            # get next argument (End of While)
            [arg, i] = getNextArg(options, i)

    # Initialize Lambda
    if l is None:
        l = 0.01 # default initial guess


    # initialize
    # ---------------------------
    #   n: Size of Inputs               [x1 .. xn]
    #   m: Size of Bases                [h1 .. hm]
    #   M: Size of all Candidate basis  [f1 .. fM], M>>m
    #   p: Number of Training Samples   {X,y}p
    #
    #   F[p,M>>m]--> H    Y[p,n]      l[1,M]

    l = np.atleast_2d(l)    # atleast_2d makes 1D to [1,n]

    [p1, M]  = F.shape
    [p2, n]  = Y.shape
    [l1, l2] = l.shape

    if p1 != p2:
        raise AssertionError('localRidge: inconsistent design matrix and training outputs')
    else:
        p = p1

    if l1 == 1:
        if l2 == 1:
            l = l * np.ones((1,M))
        elif l2 != M:
            raise AssertionError('localRidge: lambda list inconsistent length')
    elif l2 == 1:
        if l1 != M:
            raise AssertionError('localRidge: lambda list inconsistent length')
        else:
            l = l.conj().T

    # for Python l = [M,.]
    l = np.ravel(l)

    if Verbose:
      print('%s Local Ridge %s' %('\n','\n'))

    if Flops:
        flops= 0
        ##flops(0)



    # (Sections 6.3     : Eq. 6.7 - 6.9)
    # (Sections 4.1- 4.4: Eq. 4.4 - 4.10)
    # calculate model selection score
    # ---------------------------
    #   F[p,M>>m]--> H    Y[p,n]      l[1,M]

    Done   = False
    count  = 0
    keep   = np.isfinite(l)  # Pick finite Lambda value

    if Flops: print('H...')
    H = F[:,keep]                               # H [p,M-]

    if Flops: print('HH...')
    HH = np.dot(H.conj().T , H)                 # H'H [M-,M-]

    if Flops: print('A...')
    A = np.linalg.inv( HH + np.diag(l[keep]) )  # A [M-,M-] {PDF A^-1 -->Code A}

    if Flops: print('HA...')
    HA = np.dot(H , A)                          # HA [p,M-]

    if Flops: print('W...')
    W = np.dot( HA.conj().T , Y)                # W [M-,n]

    if Flops: print('P...')
    P = np.eye(p) - np.dot(HA , H.conj().T)     # P [p,p]

    if Flops: print('PY...')
    PY = np.dot(P , Y)                          # PY [p,n]

    if Flops: print('trP...')
    trP = np.trace(P)                           # trP [.]

    # (@ Section 5.2: Eq. 5.1-5.6 @)
    # set error calculation based on model
    if Flops: print('V...')
    if Term  in ['g','u','f','b']:
        YPY = traceProduct(PY.conj().T, PY)         # YPY[.]   = trace(PY' * PY)        ~ S^

        if Term == 'g':
            old_V = p * YPY / trP**2

        elif Term == 'u':
            old_V = 1 * YPY / trP

        elif Term == 'f':
            old_V = (2 * p - trP) * YPY / (p * trP)

        elif Term == 'b':
            old_V = (p + (log(p)-1) * (p-trP)) * YPY / (p * trP)
    else:
        # special case of LOO
        dPPY = PY / dupCol(np.diag(P), n)
        old_V = traceProduct(dPPY.conj().T, dPPY) / p



    if Flops: print('(%d)', flops)
    if Verbose:
        estr ='pass   in  out '
        if Term == 'u':
            estr += '   UEV    '
        else:
            estr += '   GCV    '
        estr += ' change '
        if Flops:
            estr += '  flops '
        print estr

        estr = '%4i %4i %4i %9.3e       - ' %(0, len(keep), M-len(keep), old_V)
        if Flops:
            estr += '%8i' %flops
        print estr

    # ---------------------------
    # outer loop
    # ---------------------------
    while not Done:
        # next iteration
        count += 1
        ##print "\n\n ###### COUNT %i ######"    %count

        # obtain order in which to update lambdas
        if Random:
            # random order
            ind_list = np.random.permutation(M)
        else:
            # same order every time (left to right)
            ind_list = np.arange(0,M)

        # ---------------------------
        # inner loop
        # ---------------------------
        num_in  = 0
        num_out = 0

        for k in ind_list:
            ##print "\n   ^^^^^^ k %i ^^^^^^"    %k

            # optimize this center (k) {PDF A^-1 -->Code A}
            #   F[p,m]  l[1,M]  A[m,m]  HA[p,m]    W[m,n]  PY[p,n]    P[p,p], H[p,m]
            [V, lk, A, HA, W, PY, trP] = localRidge_OptJ(k, F, l, A, HA, W, PY, trP, Term)

            # update
            l[k] = lk


            # count changes
            if lk == np.inf:
                num_out += 1
            else:
                num_in  += 1
        # ---------------------------


        # Done yet?
        # Check completion criteria
        # ---------------------------
        if count >= Hard:
            # Hard limit of iterations reached
            if Verbose:
                estr = '%4i %4i %4i %9.3e %7i ' %(count, num_in, num_out, V, change)
                if Flops:
                    estr += '%8i' %flops
                print estr
                print('hard limit reached')
            Done = True

        elif Threshold > 1:
            # use relative change in score threshold
            change = np.round(old_V / (old_V - V))
            if Verbose:
                estr = '%4i %4i %4i %9.3e %7i ' %(count, num_in, num_out, V, change)
                if Flops:
                    estr += '%8i' %flops
                print estr

            if change > Threshold:
                Done = True
                if Verbose:
                    print('relative threshold crossed')
            else:
                old_V = V

        else:
            # use absolute change in score threshold
            change = old_V - V
            if Verbose:
                estr = '%4i %4i %4i %9.3e %7i ' %(count, num_in, num_out, V, change)
                if Flops:
                    estr += '%8i' %flops
                print estr

            if change < Threshold:
                Done = True
                if Verbose:
                    print('absolute threshold crossed')
                else:
                  old_V = V

    # outputs
    # ---------------------------
    subset  = np.isfinite(l)
    H       = F[:,subset]
    P       = np.eye(p) - H.dot(A).dot(H.conj().T)

    # {PDF A^-1 -->Code A}
    return [l, V, H, A, W, P]





#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#===============================================================================
#           Optimize Single Local Parameter (Local Ridge)
#===============================================================================
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def localRidge_OptJ(k, F, l, A, HA, W, PY, trP, ms):
    """
    # [V, lp, A, HA, W, PY, trP] = localRidge_OptJ(k, F, l, A, HA, W, PY, trP, ms)
    #
    # PDF Sections A.11
    # Optimizes one local regularization parameter (l[k] --> lp).
    # l[.] = inf means local regularization parameter is inactive.
    #
    #
    #       l vector, l[k] = lj passed as input (Never Updated)
    #       lp @ min score      passed as output
    #
    #
    # The large amount of input and output variables is a result of
    # trying to minimize the number of computations by keeping
    # useful quantities around and avoiding duplicate work.
    #
    #   n: Size of Inputs               [x1 .. xn]
    #   m: Size of Bases                [h1 .. hm]
    #   M: Size of all Candidate Bases  [f1 .. fM], M>>m
    #   p: Number of Training Samples   {X,y}p
    #
    # Inputs
    #
    #  k       center index                             (int)
    #  F       Full design matrix                       (p-by-M)
    #  l       regularization parameter values          (1-by-M)
    #  A       Partial covariance matrix                (m-by-m)        {PDF A^-1 -->Code A}
    #  HA      Partial product of H and A               (p-by-m)
    #  W       Partial optimal weight matrix            (m-by-n)
    #  PY      Projected output values (y^-f)           (p-by-n)
    #  trP     Trace of the projection matrix           (1-by-1)
    #  ms      Model selection criterion                ('g' or 'u')
    #
    # Outputs
    #
    #  V       New value of GCV Sigma2
    #  lj      New j-th lamba regularization parameter  (real)
    #  A       New partial covariance matrix (A^-1)     (m-by-m)        {PDF A^-1 -->Code A}
    #  HA      New partial product of H and A           (p-by-m)
    #  W       New partial optimal weight matrix        (m-by-n)
    #  PY      New projected output values (y^-f)       (p-by-n)
    #  trP     New trace of the projection matrix       (1-by-1)
    #
    """

    #   n: Size of Inputs               [x1 .. xn]
    #   m: Size of Bases                [h1 .. hm]
    #   M: Size of all Candidate Bases  [f1 .. fM], M>>m
    #   p: Number of Training Samples   {X,y}p
    #
    #   F[p,m]  l[1,M]  A[m,m]  HA[p,m]    W[m,n]  PY[p,n]    P[p,p], H[p,m]
    # get m (current reduced size) and p
    [m, n] = W.shape
    [p, n] = PY.shape



    # -------------------------------------
    # Part 1: prepare for coefficients
    # l=inf : Inactive
    # -------------------------------------
    if l[k] < np.inf:
        # (@ Section A.11, A.7.2 @) : Remove a basis
        # (@ Ref [19] Section 10.3 Pg 9@)
        # currently active (l[k] <> inf)
        j         = np.sum( np.isfinite(l[0:k+1]) )  - 1            # NO +++1,  Column index of selected center/hj ???    l(1:k)
        j         = np.max([j,0])                                   # Make sure j>=0 if no finite lamda
        lj      = l[k]
        Dj      = 1.0 / A[j,j]                                      # A[j,j] = hj*hj + lj   {PDF A^-1 -->Code A}
        eps     = +1e-16
        Dj      = np.sign(Dj)*np.max([np.abs(Dj),eps])

        ##print "     j=%i"    %(j)
        ##print "     Dj=%s"   %(Dj)


        hjPjhj    = Dj - lj                                         # OK, stability check   [1,1]   Dj>0   Dj = lj+hjPjhj
        Pjhj      = Dj * HA[:,j][:,np.newaxis]                      # OK, implicit use      [p,1]   Ref[19], Eq 19
        hjPj2hj   = np.dot( Pjhj.conj().T , Pjhj)                   # OK, used in c,beta    [1,1]

        hjPjY     = Dj * W[j,:][np.newaxis,:]                       # OK, used in b,c       [1,n]   Eq 4.5: W=Y'.H.A^-1, Y'.Pjhj=Dj.Y'HA^-1
        PjY       = PY + np.dot(Pjhj , (hjPjY / Dj))                # OK, implicit use      [p,n] = [p,1][1,n]  Ref[19], Eq 20
        YPj2Y     = traceProduct(PjY.conj().T, PjY)                 # OK, used in a         [1,1] = trace([n,1][1,n])
        hjPj2Y    = np.dot(     Pjhj.conj().T, PjY)                 # OK, used in b         [1,n] = [1,p][p,n]
        trPj      = trP + hjPj2hj / Dj                              # OK, used in alpha     [1,1]   Ref[19], Eq 21

    else:
        # (@ Section A.11, A.7.1 @) : Add a new basis
        # previously dropped (l[k] = inf)
        actv      = np.isfinite(l)                                  # Number of Active lambdas
        j         = np.sum( np.isfinite(l[0:k+1]) ) + 1  - 1        # With +++1,Column index of selected center/hj ???    l(1:k)
        ##print "     j=%i                    **inf"    %(j)

        m        += 1
        hj        = F[:,k][:,np.newaxis]                            # new basis hj          [p,1]
        Hj        = F[:,actv]                                       # current set of bases  [p,m]
        if np.ndim(Hj)==1:
            Hj= Hj[:,np.newaxis]

        AjHj      = HA.conj().T                                     # OK, implicit use      [m,p]
        AjHjhj    = np.dot(AjHj ,     hj)                           # OK, implicit use      [m,1] = [m,p][p,1]
        HjAjHjhj  = np.dot(  Hj , AjHjhj)                           # OK, implicit use      [p,1] = [p,m][m,1]
        Pjhj      = hj - HjAjHjhj                                   # OK, implicit use      [p,1]   Pj*hj=(I-HjAjHj)*hj
        hjPjhj    = np.dot(  hj.conj().T, Pjhj)                     # OK, stability check   [1,1] = [1,p][p,1]  Dj>0
        hjPj2hj   = np.dot(Pjhj.conj().T, Pjhj)                     # OK, used in c,beta    [1,1] = [1,p][p,1]

        PjY       = PY                                              # OK, implicit use      [p,n]
        hjPjY     = np.dot(       hj.conj().T, PjY)                 # OK, used in b,c       [1,n] = [1,p][p,n]
        YPj2Y     = traceProduct(PjY.conj().T, PjY)                 # OK, used in a         [1,1] = trace([n,p][p,n])
        hjPj2Y    = np.dot(     Pjhj.conj().T, PjY)                 # OK, used in b         [1,n] = [1,p][p,n]
        trPj      = trP                                             # OK, used in alpha     [1,1]




    # (@ Section A.11: Pg. 58 @)
    # (@ Ref[19] 10.2 @)
    # get the SCALAR coefficients
    # ---------------------------
    #   F[p,m]  l[1,M]   A[m,m]  HA[p,m] W[m,n]  PY[p,n]    P[p,p]  Y[p,n]
    a       = np.asscalar( YPj2Y)                                               # Scalar trace([p,n][p,p][n,p])
    b       = np.asscalar( np.dot( hjPj2Y  , hjPjY.conj().T ) )                 # Scalar  [1,n][n,1]
    c       = np.asscalar( hjPj2hj.dot(hjPjY).dot(hjPjY.conj().T) )             # Scalar  [1,1][1,n][n,1]
    alpha   = np.asscalar( trPj )                                               # Scalar  trace(.)
    beta    = np.asscalar( hjPj2hj )                                            # Scalar [1,p][p,p][p,1]


    # -------------------------------------
    # Part 2: optimization
    #   get best min score V (sigma2 Variance)
    #   and lj there (lp = lj @ min V) and Dp
    # -------------------------------------
    if hjPjhj < 0:
        """
        Guess about source of hjPjhj < 0
        Dj = lj +hjPjhj, lj>0
        if hjPjhj<0 --> Dj can be zero and unstable!!!
        Remove basis by setting lj, Dj--> Inf
        """
        # annoyingly common numerical error - cope by not including center
        lp = np.inf

        # Calculate Sigma2 (V)
        if ms == 'g':
            V = p * a / alpha**2        # (@ Pg 58 Delta_j=Inf)
        elif ms == 'u':
            V = 1 * a / alpha
        else:
            raise NotImplementedError('localRidge_OptJ: only GCV and UEV are implemented!')

    else:
        # Case of hjPjhj >= 0
        # decide how to optimize
        # Dp = Delta_j = lj +hjPjhj (@ Section A.11)

        if ms == 'g':
            # (@ Section A.11, Pg 59 lambda j)
            # generalized cross validation (GCV)
            # V = p * (a * Dp**2 - 2 * b * Dp + c) / (alpha * Dp - beta)**2
            # Derivative(Sigma,lj)=0 --> Linear Equation Dj
            #   --> (b.alpha-a.beta)*Dj-(c.alpha-b.beta)=0
            if b*alpha == a*beta:
                lp = np.inf
            else:
                Dp = (c*alpha - b*beta) / (b*alpha - a*beta)
                lp = Dp - hjPjhj                                                # (@ Pg 59 lambda j)
                if lp < 0:
                    # check out lp = 0 and lp = Inf
                    if b*alpha > a*beta:
                        lp = 0
                        Dp = hjPjhj
                    else:
                        lp = np.inf

            # Calculate Sigma2 (V)
            if lp == np.inf:
                V = p*a / alpha**2                                          # (@ Pg 58 Delta_j=Inf)
            else:
                V = p* (a*Dp**2 - 2*b*Dp + c) / (alpha*Dp - beta)**2        # alpha,beta, Dp: scalar

        elif ms == 'u':
            # unbiased estimate of variance (UEV)
            """ NOT SURE if UEV Calculation is correct"""
            # V = (a * Dp**2 - 2 * b * Dp + c) / (Dp * (alpha * Dp - beta))
            # Derivative(Sigma,lj)=0 --> Quadratic Equation Dj
            #   --> 0.5*(2b.alpha-a.beta)*Dj^2-(c.alpha)*Dj+0.5(c.beta)=0
            if 2* b*alpha == a*beta:
                Dp = beta / (2*alpha)
            else:
                Dp = ( c*alpha + np.sqrt(c**2 * alpha**2 - c*beta * (2*b*alpha - a*beta)) ) \
                 / (2*b*alpha - a*beta)

            lp = Dp - hjPjhj

            if lp < 0:
                # check out lp=0 and lp=Inf
                V0 = (a * hjPjhj**2 - 2* b * hjPjhj + c) / (hjPjhj * (alpha* hjPjhj - beta) )
                V1 = a / alpha
                if V0 < V1:
                    lp = 0
                    V = V0
                else:
                    lp = np.inf
                    V = V1
            else:
                V = (a* Dp**2 - 2* b*Dp + c) / (Dp* (alpha*Dp - beta) )

        else:
            raise  NotImplementedError('localRidge_OptJ: only GCV and UEV implemented!')



    # -------------------------------------
    # Part 3: update for next optimization
    # with min V (sigma2 Variance)
    # and lp (lj @min V) are identified
    # -------------------------------------
    if l[k] != lp:
        ##print "     lp %2.3f <> %2.3f l[k]"    %(lp,l[k])
        ##print "     A ", A.shape

        # make changes (Remove old basis)
        # if optimized lp is different than current lamba

        # set index ranges
        ind_j = np.arange(m)
        ind_j = np.delete(ind_j, j)         # [1:j-1 j+1:m]

        # ---------------------------------
        # Calculate Aj, HjAj, W=A^-1.H'.Y
        # PjY, trPj Calculated in Part 1
        # ---------------------------------
        #   A[m,m]  HA[p,m] W[m,n]  PY[p,n] P[p,p]  H[p,m]
        if l[k] < np.inf:
            # (@ Ref [19] Section 10.3 Pg 9@)
            # currently active (l[k] <> Inf)
            # update A, HA, W stay the same [m-1,m-1] [p,m-1] [m-1,n]

            # A stuff [m-1,m-1]  {PDF A^-1 -->Code A}
            AjHjhj = -Dj * A[ ind_j ,   j   ][:,np.newaxis]                     # Ref[19], Eq Pre-19 aj=-1/Dj*Aj^-1.Hj'.hj  [m-1,1]
            hjHjAj = -Dj * A[   j   , ind_j ][np.newaxis,:]                     # Ref[19], Eq Pre-19 aj=-1/Dj*Aj^-1.Hj'.hj  [1,m-1]
            AjHjhjhjHjAj = np.dot(AjHjhj , hjHjAj)                              # OK                                        [m-1,m-1]
            Aj = A[ind_j,:][:,ind_j] - AjHjhjhjHjAj / Dj                        # Ref[19], Eq 15                            [m-1,m-1]

            # HA stuff [p,m-1]
            PjhjhjHjAj = np.dot(Pjhj , hjHjAj)                                  # OK                                        [p,m-1] = [p,1][1,m-1]
            HjAj = HA[:,ind_j] + PjhjhjHjAj / Dj                                # Hj*  Aj=A[]- AjHj.hjhjHjAj / Dj           [p,m-1]

            # W stuff [m-1,n]
            AjHjY = W[ind_j,:] + np.dot(AjHjhj , (hjPjY / Dj))                  # AjHjhj*hjPjY =-Dj*A[ind_j,j]* Dj*W[j,:]   [m-1,n] = [m-1,1][1,n]



        else:
            # previously dropped  (l[k]=Inf)
            # A, HA, W stay the same [m,m] [p,m] [m,n]

            # A stuff   {PDF A^-1 -->Code A}
            hjHjAj = AjHjhj.conj().T                                            # OK, AjHjhj is known                       [1,m]
            AjHjhjhjHjAj = np.dot(AjHjhj , hjHjAj)                              # OK                                        [m,m]
            Aj = A                                                              # OK                                        [m,m]

            # HA stuff
            PjhjhjHjAj = np.dot(Pjhj , hjHjAj)                                  # OK, Pjhj is known                         [p,m] = [p,1][1,m]
            HjAj = HA                                                           # OK                                        [p,m]

            # W stuff
            AjHjY = W                                                           # OK, W=A^-1.H'.Y                           [m,n]


        # ---------------------------------
        # Update A,HA,W,..with optimized lp
        # ---------------------------------
        #   A[m,m]  HA[p,m] W[m,n]  PY[p,n] P[p,p]  H[p,m]
        if lp < np.inf:
            # (l[k] <> lp ==> l[k]=inf)
            # (@ Section A.7.1 Eq. A.4 @)
            # update A [m,m]
            A1 = np.insert(Aj, j, 0, axis=0)                                    # insert a new row of zeros at index j      Aj [m-1,m-1]%[m,m]
            A1 = np.insert(A1, j, 0, axis=1)                                    # insert a new col of zeros at index j      A1 [m,m]%[m+1,m+1]
            ##A1 = np.delete(A1,-1,    axis=0)                                    # remove extra row from end
            ##A1 = np.delete(A1,-1,    axis=1)                                    # remove extra col from end                 A1[m,m]

            # make sure to insert new row & col first
            # then assing updated AjHjhj
            A2 = np.insert(AjHjhjhjHjAj , j, 0, axis=0)                         # insert a new row of zeros at index j      AjHjhjhjHjAj [m-1,m-1]%[m,m]
            A2 = np.insert(A2           , j, 0, axis=1)                         # insert a new col of zeros at index j      A2           [m,m]%[m+1,m+1]
            ##A2 = np.delete(A2           ,-1,    axis=0)                         # remove extra row from end
            ##A2 = np.delete(A2           ,-1,    axis=1)                         # remove extra col from end                 A2          [m,m]

            AjHjhj = np.insert(AjHjhj, j, -1, axis=0)                           # insert a new element of -1 at index j     AjHjhj       [m,1]%[m+1,1]
            ##AjHjhj = np.delete(AjHjhj,-1,     axis=0)                           # remove extra element from end
            A2[j,:]   = -AjHjhj.ravel()                                         # [m,m]/[m+1,m+1]
            A2[:,j]   = -AjHjhj.ravel()

            A = A1 + A2/Dp                                                      # [m,m]/[m+1,m+1]

            ##print "     updating j %i-->%i"    %(j,j+1)
            ##print "     A ", A.shape


            # update HA [p,m]
            # (@ Ref[19] Eq.19 * H @), Pj=1-HjAHj, Hj orthogonal to other Hm
            HA1 = np.insert(HjAj        , j, 0    , axis=1)                     # insert a new col of zeros  at index j     HjAj        [p,m]%[p,m+1]
            ##HA1 = np.delete(HA1         ,-1,        axis=1)                     # remove extra col from end
            HA2 = np.insert(PjhjhjHjAj  , j, -Pjhj.ravel(), axis=1)             # Pjhj [p,1],                               PjhjhjHjAj  [p,m]%[p,m+1]
            ##HA2 = np.delete(HA2         ,-1,        axis=1)                     # remove extra col from end
            HA  = HA1 - HA2/Dp                                                  #                                           HA          [p,m]%[p,m+1]


            # update W [m,n]
            # W = A^-1.H'.Y=Y'.H.A^-1 , Use HA update & (@ Section A.7.1 Eq. A.4 @)
            W1 = np.insert(AjHjY        , j, 0    , axis=0)                     # insert a new row of zeros at index j      AjHjY       [m,n]%[m+1,n]
            ##W1 = np.delete(W1           ,-1,        axis=0)                     # remove extra row from end
            W2 = AjHjhj                                                         # Already set above in updating A           AjHjhj      [m,1]%[m+1,1]
            W  = W1 - np.dot(W2, hjPjY)/Dp                                      # [m,n]%[m+1,n] = [m,1][1,n]

            # update Py,  Ref[19], Eq 20 (switch sides)
            PY = PjY - np.dot(Pjhj ,hjPjY)/ Dp                                  # [p,n] = [p,1][1,n]

            # update trP,  Ref[19], Eq 21 (switch sides)
            trP = trPj - hjPj2hj / Dp                                           # [1,1]


        else:
            # No update, Outputs stay the same
            # lp=Inf,  (l[k] <> lp ==> l[k]<inf)
            # Use calculated values (Aj,HjAj,AjHjY) in Part 3
            # PjY, trPj already calculated in Part 1
            A   = Aj                                                            # [m-1,m-1]
            HA  = HjAj                                                          # [p,m-1]
            W   = AjHjY                                                         # [m-1,n]
            PY  = PjY                                                           # [p,n]
            trP = trPj                                                          # [1,1]


        ##print "     Aj", Aj.shape

    # localRidge_J(k, F, l, A, HA, W, PY, trP, ms)  {PDF A^-1 -->Code A}
    return [V, lp, A, HA, W, PY, trP]
# ==============================================================================
#                              MAIN RUN PART
# ==============================================================================
if __name__ == '__main__':
    pass
