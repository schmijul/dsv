import numpy as np

#
#  Function: mse = avgSquareError(a,b)
#
#  Python version of MATLAB avgSquareError function
#
#  Input:
#       a,b - input vectors as list
#  Output:
#       mse - mean square error estimate
def avgSquareError(a,b):

    Na=a.size
    Nb=b.size
    N=np.minimum(Na,Nb)
    a=a[0:N]
    b=b[0:N]

    r=np.correlate(a,b,"full")
    lags=np.arange(-(N-1),N)

    ar = np.abs(r)
    armax   = ar.max(0)
    shiftlst = np.nonzero(armax==ar)       # list
    shift = lags[shiftlst[0][0]]           # scalar element

    if shift > 0:
        a = a[shift :]
        b = b[:-shift]
    elif shift < 0:
        a = a[0:shift]
        b = b[-shift:]
    ab=np.dot(a,b)
    bb=np.dot(b,b)
    scale=ab/bb


    d = np.abs(a-scale*b)       # abs differece
    d2 = [x**2 for x in d]      # power of 2
    mse = np.average(d**2)      # OK


    return mse

# Testbench

# a,b only for test purpose
if __name__ == '__main__':
    b=np.random.randn(30000)
    a=np.random.randn(30000)

    print(avgSquareError(a,b))
