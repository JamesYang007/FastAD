import numpy as np
from scipy.stats import bernoulli

scl_x = 0
scl_p = 0.0001

vec_x = np.array([1, 0, 1])
vec_p = np.array([0.3, 0.42, 0.98])

def logpmf(x, p):
    return np.sum(bernoulli.logpmf(x, p))

def ss_feval():
    return [logpmf(scl_x, scl_p)]

def ss_x_one_feval():
    return [logpmf(1, scl_p)]

def ss_beval():
    if scl_x == 1:
        adj =  1./scl_p
    if scl_x == 0:
        adj = -1./(1-scl_p)
    return [adj]

def ss_x_one_beval():
    adj =  1./scl_p
    return [adj]

def vs_feval():
    return [logpmf(vec_x, scl_p)]

def vs_beval():
    dp = (np.sum(vec_x) - len(vec_x) * scl_p) / (scl_p * (1-scl_p))
    return [dp]

def vv_feval():
    return [logpmf(vec_x, vec_p)]

def vv_beval():
    def adj(x, p):
        if x == 1:
            return 1/p
        if x == 0:
            return -1./(1-p)
    dp = np.array([adj(x, p) for x, p in zip(vec_x, vec_p)])
    return dp

if __name__ == '__main__':
    res = ss_x_one_beval()

    for r in res:
        print('{0:.20f}'.format(r))
