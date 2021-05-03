import numpy as np
from .kernels import kernel_vec, Kernel_Matrix
from numpy.linalg import inv


def MCMD(x, xp, y, yp, lamb = 0.01 , lambp = 0.01 ):

    n = len(x)
    m = len(xp)

    k_y = kernel_vec(y)
    K_y = Kernel_Matrix(y, y)
    W_y = inv(K_y + lamb * np.eye(n))
    K_x = Kernel_Matrix(x, x)

    k_yp = kernel_vec(yp)
    K_yp = Kernel_Matrix(yp, yp)
    W_yp = inv(K_yp + lambp * np.eye(m))
    K_xxp = Kernel_Matrix(x, xp)

    K_xp = Kernel_Matrix(xp, xp)

    return lambda y: k_y(y).T@W_y@K_x@W_y.T@k_y(y) \
                     - 2 * k_y(y).T@W_y @ K_xxp @ W_yp.T @k_yp(y) \
                     + k_yp(y).T@W_yp @K_xp @W_yp.T @k_yp(y)



def HSCIC(X, Y, Z, lamb = 0.01):

    n = len(X)
    k_z = kernel_vec(Z)
    K_z = Kernel_Matrix(Z, Z)
    K_x = Kernel_Matrix(X, X)
    K_y = Kernel_Matrix(Y, Y)

    W = inv(K_z + lamb * np.eye(n))

    return lambda z: k_z(z).T@W@ (K_x * K_y) @ W.T@ k_z(z) \
                - 2*(k_z(z).T@W@((K_x@W.T@k_z(z)) * (K_y@W.T@k_z(z))))\
                + (k_z(z).T@W@K_x@W.T@k_z(z)) * (k_z(z).T@W@K_y@W.T@k_z(z))
