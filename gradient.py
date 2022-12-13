import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def grad_J(u, q, gamma):
    return q + (2*gamma/N) * u

def B(v,w):
    res = 0
    for i in range(len(v)):
        for j in range(len(w)):
            res += np.linalg.norm(w[i] - v[j])**2
    res /= 1/(2*len(v)**2)
    return res 

def a(r, K=1, beta=2):
    return K/((1+r**2)**beta)

def a_prime(r, K=1, beta=2):
    return -2 * beta * K * r * (r**2 + 1)**(-beta - 1)

def original_system(X, t, u):
    N = int(len(X)/2)
    x = X[:N]
    v = X[N:]
    dxdt = v
    dvdt = np.zeros(len(v))
    for i in range(N):
        for j in range(N):
            dvdt += a(np.linalg.norm(x[i] - x[j]))*(v[j]-v[i])
        dvdt /= 1/N
        dvdt += u[i]
    return np.concatenate([dxdt, dvdt])

def adjoint_system(X, t, u):
    N = int(len(X)/2)
    x = X[:N]
    v = X[N:]
    dpdt = np.zeros(N)
    dqdt = np.zeros(N)
    v_bar = np.mean(v)
    for i in range(N):
        for j in range(N):
            norm_xij = np.linalg.norm(x[i] - x)
            dpdt = (a_prime(norm_xij)/norm_xij) * np.inner(q[j] - q[i], v[j] - v[i]) * (x[j] - x[i])
            dqdt = a(norm_xij) * (q[j] - q[i])

        dpdt /= 1/N
        dqdt /= 1/N

        dqdt += -2/N * (v_bar - v[i])

        dpdt *= -1
        dqdt *= -1

    return np.concatenate([dpdt, dqdt])


# Beginning of the problem
N = 10
t = np.arange(0, 10, .1)
x0 = np.random.randn(N)
v0 = np.random.randn(N)

gamma = 1

tol = 1e-3 
kmax = 100

k = 0
q = np.ones_like(N) * 100
u_cur, u_prev = np.zeros(N), np.ones(N)

while np.linalg.norm(grad_J(u_cur, q, gamma)) > tol and k < kmax:
    X = odeint(original_system, np.concatenate([x0, v0]), t, args=(u_cur,), full_output=1)[0]
    