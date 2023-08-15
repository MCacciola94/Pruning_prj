import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

def spr(w,alpha,M):
    norm2 = la.norm(w)
    norm_inf = la.norm(w,np.inf)
    const = np.sqrt(alpha/(1-alpha))

    aux = max(norm_inf/M,norm2*const)
    # breakpoint()
    if abs(norm_inf/M-norm2*const)<1e-6:
        print('border')

    if aux>1:
        print('case3')
        return norm2**2 * alpha + 1-alpha
    elif aux == norm_inf/M:
        print('case2')
        return alpha* norm2**2 *M/norm_inf + (1-alpha) * norm_inf/M
    else:
        print('case1')
        return 2*np.sqrt(alpha*(1-alpha)) *norm2
    

def check_convexity(v1,v2,alpha,M,lamb=0.5):
    val1 = spr(v1,alpha,M)
    val2 = spr(v2,alpha,M)
    val3 = spr(lamb*v1+(1-lamb)*v2,alpha,M)
    val4 = lamb*val1+(1-lamb)*val2

    print(val1)
    print(val2)
    print(val3)
    print(val4)

    print("Is convex? ", val3<=val4)
    return val3<=val4


def try_conv(v1,v2,alpha,M):
    for i in range(100):
        l= (i+1)*0.0095
        if not check_convexity(v1,v2,alpha,M,lamb=l):
            print('lamb ',l)
            break


def plot(v1,v2,alpha,M):
        x = [i*0.01 for i in range(101)]
        y = [spr(v1*l+(1-l)*v2,alpha,M) for l in x]

        plt.plot(x,y)
        plt.show()
        plt.savefig('spr_try.png')