"""
Metropolis simulation of the classical Ising model on a square lattice. 
Algorithm:
1) Create a random N x N matrix of 1 and -1 (initial configuration)
2) Select one spin (site) randomly 
3) Generate random number r between 0 and 1
4) If r< \exp[-\Delta E/\beta] (Boltzmann weight) flip the spin
5) Select another random spin (also the same) and go to 3
---------------------------------------------------------------
Do this until you reach "equilibrium": the averages don't change
"""
#%%
import numpy as np
import matplotlib.pyplot as plt 
import time
#%%
def initialConf(N):
    #function that generates a N x N random matrix of 1 and  -1
    mylist=np.array([-1,1])
    y=mylist[np.random.randint(len(mylist), size=(N,N))]
    return y

def energy(J, H, k,l, conf, flip):
    #function that computes the energy of a certain 
    #configuration with the spin (k,l) flipped or not
    #J: Ising interaction, H: local field
    #conf: current configuration N x N matrix
    #flip= False (not flipped) True (flipped)
    nrow=len(conf)
    ncol=len(conf[0])
    if flip==False:
        x=conf[k,l]
    elif flip==True:
        x=conf[k,l]-2*np.sign(conf[k,l])
    if 0<k<nrow-1 and 0<l<ncol-1:
        y=-H*x-J*x*(conf[k,l+1]+conf[k,l-1]+conf[k-1,l]+conf[k+1,l])
    #boundaries (periodic)
    elif k==0 and 0<l<ncol-1:
        y=-H*x-J*x*(conf[k,l+1]+conf[k,l-1]+conf[nrow-1,l]+conf[k+1,l])
    elif k==nrow-1 and 0<l<ncol-1:
        y=-H*x-J*x*(conf[k,l+1]+conf[k,l-1]+conf[k-1,l]+conf[0,l])
    elif 0<k<nrow-1 and l==0:
        y=-H*x-J*x*(conf[k,l+1]+conf[k,ncol-1]+conf[k-1,l]+conf[k+1,l])
    elif 0<k<nrow-1 and l==ncol-1:
        y=-H*x-J*x*(conf[k,0]+conf[k,l-1]+conf[k-1,l]+conf[k+1,l])
    #corners
    elif k==0 and l==0:
        y=-H*x-J*x*(conf[k,l+1]+conf[k,ncol-1]+conf[nrow-1,l]+conf[k+1,l])
    elif k==0 and l==ncol-1:
        y=-H*x-J*x*(conf[k,0]+conf[k,l-1]+conf[nrow-1,l]+conf[k+1,l])
    elif k==nrow-1 and l==0:
        y=-H*x-J*x*(conf[k,l+1]+conf[k,ncol-1]+conf[k-1,l]+conf[0,l])
    elif k==nrow-1 and l==ncol-1:
        y=-H*x-J*x*(conf[k,0]+conf[k,l-1]+conf[k-1,l]+conf[0,l])
    return y
def magnetization(conf):
    #returns the modulus of the magnetization per unit of spin
    y=np.abs(np.sum(conf))/np.size(conf)
    return y

def metropolisIt(J, H, beta, conf):
    spinList=list(range(0, np.size(conf)))
    z=spinList[np.random.randint(len(spinList))]
    k=np.floor_divide(z, N)
    l=np.mod(z,N)
    Ein=energy(J, H, k,l, conf, False)
    Efin=energy(J, H, k,l, conf, True)
    DeltaE=Efin-Ein
    boltzFactor=np.exp(-DeltaE*beta)
    r=np.random.random_sample()
    if r<boltzFactor:
        conf[k,l]=conf[k,l]-2*np.sign(conf[k,l])
    return conf

#%%
"""
Inputs
"""
J=1
H=0
beta=0.2
N=30 #N x N matrix so N^2 spins
Miter=1000*N**2 #number of iterations
#%%
conf=initialConf(N)
xIt=np.arange(0, Miter+1, 1)
magnListEq=np.zeros(len(xIt))
yMagn= magnetization(conf)
magnListEq[0]=yMagn
start=time.time()
for i in range (0, Miter):
    conf=metropolisIt(J, H, beta, conf)
    yMagn+= magnetization(conf)
    magnListEq[i+1]=yMagn/(i+2)
end=time.time()
print(end-start)


#%%
yticks=np.arange(0,1, 0.1)
fig1=plt.figure(figsize=(8,6))
plt.plot(xIt, magnListEq, linewidth=2.0, color='mediumblue')
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Magnetization', fontsize=14)
plt.title('Equilibration: $\\beta/J=$ ' +str(beta)+', $H/J=$' +str(H), fontsize=14)
plt.yticks(np.arange(0,1.1, 0.1))
plt.show()
#%%
"""
We now want to compute the phase diagram, i.e., magnetization
as a function of beta/J (H=0).
"""
start=time.time()
betaList=np.linspace(0.1, 4, 50)
magnListBeta=np.zeros(len(betaList))
for i in range(0, len(betaList)):
    conf=initialConf(N)
    magn= magnetization(conf)/(Miter+1)
    for j in range(0, Miter):
        conf=metropolisIt(J, H, betaList[i], conf)
        magn+=magnetization(conf)/(Miter+1)
    magnListBeta[i]=magn
end=time.time()
print(end-start)
#%%
fig2=plt.figure(figsize=(8,6))
plt.plot(betaList, magnListBeta, linewidth=2.0, color='teal')
plt.xlabel('$\\beta/J$', fontsize=14)
plt.ylabel('Magnetization', fontsize=14)
plt.title('Phase transition', fontsize=14)
plt.show()


#%%
