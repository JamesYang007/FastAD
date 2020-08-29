import numpy as np
from scipy.stats import norm, multivariate_normal

scl_x = 2.31
scl_mu = -0.2
scl_sigma = 0.01

vec_x = np.array([3.1, -2.3, 1.3])
vec_mu = np.array([-0.3, -2.3, -1.2])
vec_sigma = np.array([0.01, 1.03, 2.41])

mat_sigma = np.array([
    [1.0, 0.3, 0.2],
    [0.3, 2.0, -0.3],
    [0.2, -0.3, 3.0]
])

def log_pdf_adj(x, mu, s):
    if isinstance(x, float):
        n = 1
    else:
        n = len(x)

    if isinstance(s, np.ndarray) and s.shape[0] == s.shape[1]:
        return multivariate_normal.logpdf(x, mu, s) + (n/2) * np.log(2*np.pi)
    else:
        return np.sum(norm.logpdf(x, mu, s)) + (n/2) * np.log(2*np.pi)

def sss_feval():
    return [log_pdf_adj(scl_x, scl_mu, scl_sigma)]

def sss_beval():
    dx = -(scl_x - scl_mu) / scl_sigma**2
    dmu = -dx
    dsigma = (((scl_x - scl_mu)/scl_sigma)**2 - 1) / scl_sigma
    return [dx, dmu, dsigma]

def vss_feval():
    return [log_pdf_adj(vec_x, scl_mu, scl_sigma)]

def vss_beval():
    dx = -(vec_x - scl_mu) / scl_sigma**2
    dmu = np.sum(vec_x - scl_mu) / scl_sigma**2
    dsigma = (np.sum(((vec_x - scl_mu)/scl_sigma)**2) - len(vec_x)) / scl_sigma
    return np.concatenate([dx, [dmu, dsigma]])

def vvs_feval():
    return [log_pdf_adj(vec_x, vec_mu, scl_sigma)]

def vvs_beval():
    dx = -(vec_x - vec_mu) / scl_sigma**2
    dmu = -dx
    dsigma = (np.sum(((vec_x - vec_mu)/scl_sigma)**2) - len(vec_x)) / scl_sigma
    return np.concatenate([dx, dmu, [dsigma]])

def vsv_feval():
    return [log_pdf_adj(vec_x, scl_mu, vec_sigma)]

def vsv_beval():
    dx = -(vec_x - scl_mu) / vec_sigma**2
    dmu = np.sum((vec_x - scl_mu) / vec_sigma**2)
    dsigma = (((vec_x - scl_mu)/vec_sigma)**2 - 1)/vec_sigma
    return np.concatenate([dx, [dmu], dsigma])

def vvv_feval():
    return [log_pdf_adj(vec_x, vec_mu, vec_sigma)]

def vvv_beval():
    dx = -(vec_x - vec_mu) / vec_sigma**2
    dmu = -dx
    dsigma = (((vec_x - vec_mu)/vec_sigma)**2 - 1)/vec_sigma
    return np.concatenate([dx, dmu, dsigma])

def vsm_feval():
    return [log_pdf_adj(vec_x, scl_mu*np.ones(3), mat_sigma)]

def vsm_beval():
    inv = np.linalg.inv(mat_sigma)
    dx = -inv.dot(vec_x - scl_mu)
    dmu = -np.sum(dx)
    dsigma = -0.5 * (inv - np.outer(dx, np.transpose(dx)))
    return np.concatenate([dx, [dmu], (dsigma).flatten()])

def vvm_feval():
    return [log_pdf_adj(vec_x, vec_mu, mat_sigma)]

def vvm_beval():
    inv = np.linalg.inv(mat_sigma)
    dx = -inv.dot(vec_x - vec_mu)
    dmu = -dx
    dsigma = -0.5 * (inv - np.outer(dx, np.transpose(dx)))
    return np.concatenate([dx, dmu, (dsigma).flatten()])

if __name__ == "__main__":
    res = vvm_beval()
    for r in res:
        print('{0:.16f}'.format(r))
