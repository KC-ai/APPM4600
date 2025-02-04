import numpy as np
import matplotlib as plt

def fixed_point_method(g,dg,x0,a,b,tol,nmax,vrb=False):
     # Fixed point iteration method applied to find the fixed point of g from starting point x0

     # Initial values
     n=0
     xn = x0
     # Current guess is stored at rn[n]
     rn=np.array([xn])
     r=xn

     if vrb:
         print("\n Fixed point method with nmax=%d and tol=%1.1e\n" % (nmax, tol))
         print("\n|--n--|----xn----|---|g(xn)|---|---|g'(xn)---|")

     while n<=nmax:
         if vrb:
             print("|--%d--|%1.8f|%1.8f|%1.4f|" % (n,xn,np.abs(g(xn)),np.abs(dg(xn))))

         # If the estimate is approximately a root, get out of while loop
         if np.abs(g(xn)-xn)<tol:
             #(break is an instruction that gets out of the while loop)
             break

         # update iterate xn, increase n.
         n += 1
         xn = g(xn); #apply g (fixed point step)
         rn = np.append(rn,xn) #add new guess to list of iterates

     # Set root estimate to xn.
     r=xn

     if vrb:
         ########################################################################
         # Approximate error log-log plot
         logploterr(rn,r)
         plt.title('Fixed Point Iteration: Log error vs n')
         ########################################################################

     return r, rn

def logploterr(rn,r):
    n = rn.size-1
    e = np.abs(r-rn[0:n])
    #length of interval
    nn = np.arange(0,n)
    #log plot error vs iteration number
    plt.plot(nn,np.log2(e),'r--')
    plt.xlabel('n'); plt.ylabel('log2(error)')
    return

def aitken_delta_squared(seq):
    n = len(seq) - 2 
    if n < 1:
        return seq  

    faster = np.zeros(n)

    for i in range(n):
        num = (seq[i+1] - seq[i])**2
        denom = seq[i+2] - 2 * seq[i+1] + seq[i]

        if np.abs(denom) < 1e-12:  
            return faster[:i]

        faster[i] = seq[i] - num / denom

    return faster

