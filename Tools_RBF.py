#-------------------------------------------------------------------------------
# Name:        RBF Utility Tools
# Purpose:      General utilities used in RBF
#
# Author:      Mohammad
#
# Created:     04/05/2016
# Copyright:   (c) ISI 2016
# Licence:     ISI
#-------------------------------------------------------------------------------
import os, sys
import numpy as np


mypath    = os.path.dirname(os.path.realpath(sys.argv[0]))


#===============================================================================
#           Duplicae Row
#===============================================================================
def dupRow(v, m):
    """
    #  M = dupCol(v, m)
    #
    #  Duplicates v, a row vector, m times. Returns the
    #  result as matrix M with m rows, each one a copy of v.
    #
    #  Inputs
    #
    #    v    a row vector (1-by-n)
    #    m    a positive integer
    #
    #  Output
    #
    #    M    a matrix (m-by-n) matrix
    """
    if not isinstance(v, (np.ndarray, list)):
        raise AssertionError('dupCol: input must be list or array')

    vec = np.array(v)
    if vec.ndim==1:
        vec = np.atleast_2d(vec)    # Row Vector
    [r, c] = vec.shape

    if r != 1:
      raise AssertionError('dupRow: input vector must be row')

    M = np.repeat(vec,m,0)
    return M


#===============================================================================
#           Duplicae Col
#===============================================================================
def dupCol(v, n):
    """
    #  M = dupCol(v, n)
    #
    #  Duplicates v, a column vector, n times. Returns the
    #  result as matrix M with n columns, each one a copy of v.
    #
    #  Inputs
    #
    #    v    a column vector (m-by-1)
    #    n    a positive integer
    #
    #  Output
    #
    #    M    a matrix (m-by-n) matrix
    """
    if not isinstance(v, (np.ndarray, list)):
        raise AssertionError('dupCol: input must be list or array')

    vec = np.array(v)
    if vec.ndim==1:
        vec = np.atleast_2d(vec).T      # Column Vector
    [r, c] = vec.shape

    if c != 1:
      raise AssertionError('dupCol: input vector must be column')

    M = np.repeat(vec,n,axis=1)
    return M

#===============================================================================
#           Sum Row
#===============================================================================
def rowSum(X):
    """
    #  s = rowSum(X)
    #
    #  Outputs a column vector whose elements are the
    #  sums of the rows of X.
    #
    #  Inputs
    #
    #    X     matrix (m-by-n)
    #
    #  Output
    #
    #    s     vector (m-by-1)
    """

    ##[m,n]   = X.shape
    d       = X.ndim

    ##if n > 1:
    if d > 1:
    	s = np.sum(X, axis=1)[:,np.newaxis]
    else:
    	s = X[:,np.newaxis]

    return s

#===============================================================================
#           Sum Col
#===============================================================================
def colSum(X):
    """
    #  s = colSum(X)
    #
    #  Outputs a row vector whose elements are the
    #  sums of the columns of X.
    #  Designed to get round the feature of the standard
    #  routine (sum) of summimg row vectors to a scalar.
    #  If colSum is handed a row vector, the same vector
    #  is given back.
    #
    #  Inputs
    #
    #    X     matrix (m-by-n)
    #
    #  Output
    #
    #    s     vector (1-by-n)
    """
    ##[m,n]   = X.shape
    d       = X.ndim

    ##if m > 1:
    if d > 1:
    	s = np.sum(X, axis=0)[np.newaxis,:]
    else:
    	s = X[np.newaxis,:]

    return s

#===============================================================================
#           Diagonal of Product (X & Y)
#===============================================================================
def diagProduct(X, Y):
    """
    #  d = diagProduct(X, Y)
    #
    #  Outputs the diagonal of the product of X and Y.
    #  Faster than diag(X*Y).
    #
    #  Inputs
    #
    #    X    matrix (m-by-n)
    #    Y    matrix (n-by-m)
    #
    #  Output
    #    d    vector (m-by-1)
    """

    [m,n] = X.shape
    [p,q] = Y.shape

    if (m!=q) or  (n!= p):
      raise AssertionError('diagProduct: bad dimensions')
      return None


    ### P - a column vector of the rows of X  [n*m,1]
    ##P = X.conj().T
    ##P = np.ravel(P)[:,np.newaxis]
    ##
    ### Q - a column vector of the columns of Y  [n*m,1]
    ##Q = np.ravel(Y)[:,np.newaxis]
    ##
    ### Z - an [n,m] matrix containing the components of P.*Q
    ##Z = np.reshape(P * Q, (n,m))
    ##
    ### d - the answer is the sum of the columns of Z
    ##d = colSum(Z).conj().T

    d = np.sum(X*Y.conj().T,axis=-1)[:,np.newaxis]
    return d

#===============================================================================
#           Trace Product (X & Y)
#===============================================================================
def traceProduct(X, Y):
    """
    #  t = traceProduct(X, Y)
    #
    #  Outputs the trace of the product of X and Y.
    #  Faster than trace(X*Y).
    #
    #  Inputs:
    #
    #    X    matrix (m-by-n)
    #    Y    matrix (n-by-m)
    #
    #  Output:
    #
    #    t    scalar
    """
    [m,n] = X.shape
    [p,q] = Y.shape

    if (m!=q) or (n!=p):
      raise AssertionError('traceProduct: bad dimensions')
      return None


    # use the fast diagonal of a product routine
    t = np.sum( diagProduct(X, Y) );
    return t



#===============================================================================
#           overWrite
#===============================================================================
def overWrite(b, n=None):
    """
    #  b = overWrite(b, n)
    #
    #  Backup b characters (the number of characters in the
    #  string printed by the last call to {\tt overWrite})
    #  and print the new string, returning its length in {\tt b}.
    #
    #  Inputs
    #
    #    b      number of characters used in last overWrite
    #    n      string or integer
    #
    #  Outputs
    #
    #    b      number of characters used in this overWrite
    """

    # backup
    for i in range(b):
      print '\b'


    if n is None:
      # rub out
      s = ' '
      print '#s' #s(ones((1,b)))

      # backup again
      for i in range(b):
        print '\b'
    else:
      # print
      if isinstace(n,str):
        print '#s' #n
        b = len(n)
      else:
        s = '#d' #n
        print '#s' #s
        b = len(s)

    return b


#===============================================================================
#           Get Next Argument in String Option
#===============================================================================
def getNextArg(s, i):
    """
    # [arg, i] = getNextArg(s, i)
    #
    # Used to parse a comma or space separated list of arguments
    # in a single string into a sequence of argument strings.
    #
    # Inputs
    #
    #   s     entire string
    #   i     current position in string
    #
    # Outputs
    #
    #   arg   sub-string
    #   i     new position in string
    """
    # check if already at end
    if i >= len(s):
      arg = ''

    else:
      if (s[i] ==' ')  or (s[i] == ','):
        # skip over separators
        separator = True

        while separator:
          i += 1
          if i >= len(s):
            separator = False
          else:
            separator = (s[i] == ' ') or (s[i] == ',')

      # check if at the end (with +1 shift)
      if i > len(s):
        arg = ''

      else:
        # scan word
        separator = False
        arg = ''

        while not separator:
          arg += s[i]
          i += 1

          if i >= len(s):
            separator = True
          else:
            separator = (s[i] == ' ') or (s[i] == ',')

    return [arg, i]


# ==============================================================================
#                              MAIN RUN PART
# ==============================================================================
if __name__ == '__main__':
    pass
