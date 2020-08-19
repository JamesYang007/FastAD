import numpy as np
from scipy.stats import cauchy

scl_x = 0.421
scl_loc = 0.341
scl_scale = 2.132
vec_x = np.array([0.5, -1.3, -3.2414999])
vec_loc = np.array([0.4, -2.30000001, -10.32])
vec_scale = np.array([0.51, 0.01, 3.4])

def cauchy_adj_log_pdf(x, loc, scale):
    if isinstance(x, float): correction = np.log(np.pi)
    else: correction = len(x) * np.log(np.pi)
    return np.sum(cauchy.logpdf(x, loc, scale)) + correction

def sss_feval():
    return np.array([cauchy_adj_log_pdf(scl_x, scl_loc, scl_scale)])

def sss_beval():
    denom = scl_scale + (scl_x - scl_loc)**2 / scl_scale
    dx = -2*(scl_x - scl_loc) / (scl_scale * denom)
    dx0 = -dx
    dgamma = (((scl_x-scl_loc)/scl_scale)**2 - 1) / denom
    if scl_scale > 0:
        return np.array([dx, dx0, dgamma])
    else:
        return np.array([0,0,0])

def vss_feval():
    return np.array([cauchy_adj_log_pdf(vec_x, scl_loc, scl_scale)])

def vss_beval():
    diff = vec_x - scl_loc
    dx = -2. * diff / (scl_scale**2 + diff**2)
    dx0 = -np.sum(dx)
    dgamma = -np.sum((1 - (diff/scl_scale)**2)/(scl_scale + (diff**2/scl_scale)))
    return np.concatenate([dx, [dx0, dgamma]])

def vsv_feval():
    return np.array([cauchy_adj_log_pdf(vec_x, scl_loc, vec_scale)])

def vsv_beval():
    diff = vec_x - scl_loc
    dx = -2. * diff / (vec_scale**2 + diff**2)
    dx0 = -np.sum(dx)
    dgamma = -(1 - (diff/vec_scale)**2)/(vec_scale + (diff**2/vec_scale))
    return np.concatenate([dx, [dx0], dgamma])

def vvs_feval():
    return np.array([cauchy_adj_log_pdf(vec_x, vec_loc, scl_scale)])

def vvs_beval():
    diff = vec_x - vec_loc
    dx = -2. * diff / (scl_scale**2 + diff**2)
    dx0 = -dx
    dgamma = -np.sum((1 - (diff/scl_scale)**2)/(scl_scale + (diff**2/scl_scale)))
    return np.concatenate([dx, dx0, [dgamma]])

def vvv_feval():
    return np.array([cauchy_adj_log_pdf(vec_x, vec_loc, vec_scale)])

def vvv_beval():
    diff = vec_x - vec_loc
    dx = -2. * diff / (vec_scale**2 + diff**2)
    dx0 = -dx
    dgamma = -(1 - (diff/vec_scale)**2)/(vec_scale + (diff**2/vec_scale))
    return np.concatenate([dx, dx0, dgamma])

if __name__ == "__main__":
    res = vvv_beval()

    for r in res:
        print('{0:.18f}'.format(r))
