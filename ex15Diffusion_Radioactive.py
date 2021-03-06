import numpy as np
import matplotlib.pyplot as plt


# Paramètres d'intégration numérique
N=9 #nombre de points d'espace à l'intérieur
x = np.linspace(0., 1., N+2)
tmax = 0.5
# Conditions frontières
bc = [0, 0]
D = 1
b = 1
lmb = 0.5 #Lambda
#Fonction de condition initiale u(x,t=0)
U0_sin = np.sin(np.pi*x)

def tridiag(a, b, c, f):
    """Résolution système à matrice tridiagonale"""
    N = f.size
    x = np.zeros(N)
    cstar = np.zeros(N)
    astar = a[0]
    x[0] = f[0]/astar
    for k in range(1, N):
        cstar[k-1] = c[k-1]/astar
        astar = a[k]-b[k]*cstar[k-1]
        x[k] = (f[k]-b[k]*x[k-1])/astar
    for k in range(N-2, -1, -1):
        x[k] -= cstar[k]*x[k+1]
    return(x)

def EulArriere(U0, x, tmax, lmb, bc, b, D):
    """Implémentation d'Euler Arrrière"""
    #N = np.shape(x)[0]-2
    h = x[1]-x[0]
    k = lmb*h**2
    M = int(np.round(tmax/k))+1
    t = np.linspace(0., tmax, M)
    k = t[1]-t[0]
    lmb = k/h**(2)
    # np.shape renvoit un tuple le [0] permet de juste prendre le scalaire
    U = np.zeros((N+2, np.shape(t)[0]))
    U[:, 0] = U0
    a = (1+k*b+2*D*lmb)*np.ones(N)
    b = -D*lmb*np.ones(N-1)
    c = -D*lmb*np.ones(N-1)
    B = np.diagflat(a)+np.diagflat(b,-1)+np.diagflat(c,1)
    for i in range(np.shape(t)[0]-1):
        Uc = U[1:-1, i]
        U[1:-1, i+1] = np.linalg.solve(B, Uc)
        U[0, i+1] = bc[0]
        U[-1, i+1] = bc[1]
    return(U, lmb, t)

EAr_sin = EulArriere(U0_sin, x, tmax, lmb, bc, b, D)
t = EAr_sin[2]

# Solution analytique pour condition initiale carré et sinusoidal
xsol = np.linspace(0, 1, 100)
Sol_sin = np.zeros((100, np.shape(t)[0]))
for i in range(np.shape(t)[0]):
    Sol_sin[:, i] = np.sin(np.pi*xsol)*np.exp(-(b+D*(np.pi**2))*t[i])

# Plot
M = t.shape[0]
Mspan = [0, 0.25, 0.5, 0.75, 0.99]
fig, ax = plt.subplots(1,2)
for i in range(len(Mspan)):
    n = int(np.floor(M*Mspan[i]))
    ax[0].plot(xsol, Sol_sin[:,n], label ='t = {}$s$'.format(n))
    ax[0].legend(loc='best', fontsize='small')
    ax[0].set_title('Analytique')
    ax[1].plot(x, EAr_sin[0][:,n],marker='^', label ='t = {}$s$'.format(n))
    ax[1].legend(loc='best', fontsize='small')
    ax[1].set_title('Euler Arrière')


    plt.tight_layout()
    plt.suptitle('Diffusion Radioactive D ={}, b = {}, lambda ={}'.format(D,b,lmb))


plt.show()
