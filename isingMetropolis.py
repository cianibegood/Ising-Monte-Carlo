"""
Metropolis simulation of the classical Ising model on a square lattice. 
Algorithm:
1) Create a random N x N matrix of 1 and -1 (initial configuration)
2) Select one spin (site) randomly 
3) Generate random number r between 0 and 1
4) If r< \exp[-\Delta E/\beta] (Boltzmann weight) flip the spin
5) Select another spin (different) randomly and go to 3
---------------------------------------------------------------
Do this until you do it for all the spins (maybe you can do less, 
but at any rate it is efficient). Repeat the all procedure M times
and average over the final configurations.
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
def metropolisIt(J, H, beta):
    conf= initialConf(N)
    spinList=list(range(0, N**2))
    for i in range (0, N**2):
        z=spinList[np.random.randint(len(spinList))]
        k=np.floor_divide(z, N)
        l=np.mod(z,N)
        spinList.remove(z)
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
J=100
H=0
beta=100
N=5
#%%
print(metropolisIt(J, H, beta))