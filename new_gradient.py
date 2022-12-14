import numpy as np
from scipy.integrate import odeint

def grad_J(u, q, gamma):
    N = len(q)
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

def a_prime(r, K=1, beta=0):
    return -2 * beta * K * r * ((r**2 + 1)**(-beta - 1))

def f(X, t, u):
    N = int(len(X)/4)
    x = X[:N]
    v = X[N:2*N]
    p = X[2*N:3*N]
    q = X[3*N:]

    dxdt = v
    dvdt = np.zeros(N)

    dpdt = np.zeros(N)
    dqdt = np.zeros(N)

    v_bar = np.mean(v)

    for i in range(N):
        for j in range(N):
            norm_xij = np.linalg.norm(x[i] - x[j])
            if norm_xij > 0:
                dvdt[i] += a(np.linalg.norm(norm_xij))*(v[j]-v[i])
            
                dpdt[i] += (a_prime(norm_xij)/norm_xij) * np.inner(q[j] - q[i], v[j] - v[i]) * (x[j] - x[i])
                dqdt[i] += a(norm_xij) * (q[j] - q[i])
        
        dvdt[i] *= 1/N
        dvdt[i] += u[i]

        dpdt[i] *= 1/N
        dqdt[i] *= 1/N

        dqdt[i] += p[i] - (2/N) * (v_bar - v[i])

        dpdt[i] *= -1
        dqdt[i] *= -1

    return np.concatenate([dxdt, dvdt, dpdt, dqdt])



N = 10
t = np.concatenate([np.arange(0, 10, .1),[10]])
x0 = np.random.randn(N) * 1.5
v0 = np.random.randn(N) * .5
q0 = np.zeros(N)
p0 = np.zeros(N)

u_prev = np.zeros(N)
u_cur  = np.ones(N)

X0 = np.concatenate([x0,v0,p0,q0])

tol, kmax = 1e-3, 100

grad, k = 999, 0 
while np.linalg.norm(grad) > tol and k < kmax:
    sol = odeint(f,X0, t, args=(u_cur,))
    x = sol[:N]
    v = sol[N:2*N]
    p = sol[2*N:3*N]
    q = sol[3*N:]

    grad = grad_J(u_cur, q, 0.5)
    print(k, np.linalg.norm(grad))
    dif_grad = grad_J(u_cur, q, 0.5) - grad_J(u_prev, q, 0.5)
    alpha_k = np.inner(u_cur - u_prev, dif_grad) / np.linalg.norm(dif_grad)**2
    u_prev = u_cur
    u_cur -= alpha_k * grad
    k += 1 

print(k)