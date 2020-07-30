import numpy as np
from scipy.stats import wishart
from scipy.special import multigammaln

x = np.array([
    [10, 2, 3],
    [2, 10, 1],
    [3, 1, 10]
])

v = np.array([
    [5, 1, 0],
    [1, 5, 1],
    [0, 1, 5]
])

n = 4

def logpdf(x, v, n):
    correction = (n * v.shape[0] / 2.)*np.log(2) + \
                multigammaln(n/2, v.shape[0])
    return [np.sum(wishart.logpdf(x, n, v)) + correction]

def feval():
    return logpdf(x, v, n)

if __name__ == "__main__":
    res = feval()

    for r in res:
        print('{0:.20f}'.format(r))
