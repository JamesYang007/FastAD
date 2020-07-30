import numpy as np
from scipy.stats import uniform

scl_x = 0.45
scl_min = -3.2415
scl_max = 0.5231
vec_x = np.array([0.5, -2.3, -3.2414999])
vec_min = np.array([0.4, -2.30000001, -10.32])
vec_max = np.array([0.51, 0., 3.4])

def uniform_adj_log_pdf(x, m, M):
    return np.sum(uniform.logpdf(x, m, M))

def sss_feval():
    return np.array([uniform_adj_log_pdf(scl_x, scl_min, scl_max-scl_min)])

def sss_beval():
    v = 1./(scl_max - scl_min)
    if scl_min < scl_x and scl_x < scl_max:
        return np.array([v, -v])
    else:
        return np.array([0,0])

def vss_feval():
    return np.array([uniform_adj_log_pdf(vec_x, scl_min, scl_max-scl_min)])

def vss_beval():
    return len(vec_x) * sss_beval()

def vsv_feval():
    return np.array([uniform_adj_log_pdf(vec_x, scl_min, vec_max-scl_min)])

def vsv_beval():
    dM = -1./(vec_max - scl_min)
    dm = np.sum(-dM)
    return np.concatenate([[dm], dM])

def vvs_feval():
    return np.array([uniform_adj_log_pdf(vec_x, vec_min, scl_max-vec_min)])

def vvs_beval():
    dm = 1./(scl_max - vec_min)
    dM = np.sum(-dm)
    return np.concatenate([dm, [dM]])

def vvv_feval():
    return np.array([uniform_adj_log_pdf(vec_x, vec_min, vec_max-vec_min)])

def vvv_beval():
    dM = -1./(vec_max - vec_min)
    dm = -dM
    return np.concatenate([dm, dM])

if __name__ == "__main__":
    res = vvv_beval()

    for r in res:
        print('{0:.16f}'.format(r))
