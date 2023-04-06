####################################
#  CCSD ENERGY CALCULATION ON 
#    HeH+ / STO-3G / R = 0.9295 Ang
#
#  Reference for Equations:
#    Stanton, et al
#      J. Chem. Phys 94 (6),
#      15 March 1991
#
####################################

from __future__ import division
import math
import numpy as np
from itertools import product

####################################
#
#   FUNCTIONS
#
####################################

# Return Value of spatial MO two electron integral
# Example: (12\vert 34) = tei(1,2,3,4)
def teimo(a,b,c,d):
    eint = lambda x, y: x*(x+1)/2 + y if x>y else y*(y+1)/2 + x # compound index given two indices
    return ttmo.get(eint(eint(a,b),eint(c,d)),0.0e0)

####################################
#
#  INITIALIZE ORBITAL ENERGIES 
#  AND TRANSFORMED TWO ELECTRON
#  INTEGRALS  
#
####################################

Nelec = 2 # we have 2 electrons in HeH+
dim = 2 # we have two spatial basis functions in STO-3G
E = [-1.52378656, -0.26763148] # molecular orbital energies

# python dictionary containing two-electron repulsion integrals
ttmo = {5.0: 0.94542695583037617, 12.0: 0.17535895381500544, 14.0: 0.12682234020148653, 17.0: 0.59855327701641903, 19.0: -0.056821143621433257, 20.0: 0.74715464784363106}
ENUC = 1.1386276671 # nuclear repulsion energy for HeH+ -- constant
EN   = -3.99300007772 # SCF energy

####################################################
#
#  CONVERT SPATIAL TO SPIN ORBITAL MO
#
####################################################

# This makes the spin basis double bar integral (physicists' notation)
spinints=np.zeros((dim*2,dim*2,dim*2,dim*2))
gen_iter = [range(2,2*dim+2)]*4
for p, q, r, s in product(*gen_iter):
    value1 = teimo(p>>1,r>>1,q>>1,s>>1) * (p%2 == r%2) * (q%2 == s%2)
    value2 = teimo(p>>1,s>>1,q>>1,r>>1) * (p%2 == s%2) * (q%2 == r%2)
    spinints[p-2,q-2,r-2,s-2] = value1 - value2

#####################################################
#
#  Spin basis fock matrix eigenvalues 
#
#####################################################

fs = np.diag([E[i>>1] for i in range(2*dim)]) # put MO energies in diagonal array

#######################################################
#
#   CCSD CALCULATION
#
#######################################################

dim *= 2 # twice the dimension of spatial orbital

# Init empty T1 (ts) and T2 (td) arrays
ts = np.zeros((dim,dim))
td = np.zeros((dim,dim,dim,dim))

# Make denominator arrays Dai, Dabij
Dai = np.zeros((dim,dim))
Dabij = np.zeros((dim,dim,dim,dim))

gen_iter = [range(Nelec,dim),range(Nelec)]
for a, i in product(*gen_iter):
    # Equation (12) of Stanton
    Dai[a,i] = fs[i,i] - fs[a,a]
    for b, j in product(*gen_iter):
        # Initial guess T2 --- from MP2 calculation!
        td[a,b,i,j] += spinints[i,j,a,b]/(fs[i,i] + fs[j,j] - fs[a,a] - fs[b,b])
        # Stanton eq (13)
        Dabij[a,b,i,j] = fs[i,i] + fs[j,j] - fs[a,a] - fs[b,b]


# Stanton eq (9)
taus = lambda a, b, i, j: td[a,b,i,j] + 0.5*(ts[a,i]*ts[b,j] - ts[b,i]*ts[a,j])

# Stanton eq (10)
tau = lambda a, b, i, j: td[a,b,i,j] + ts[a,i]*ts[b,j] - ts[b,i]*ts[a,j]

# We need to update our intermediates at the beginning, and 
# at the end of each iteration. Each iteration provides a new
# guess at the amplitudes T1 (ts) and T2 (td), that *hopefully*
# converges to a stable, ground-state, solution.
def updateintermediates():
    Fae = np.zeros((dim,dim))
    Fmi = np.zeros((dim,dim))
    Fme = np.zeros((dim,dim))
    Wmnij = np.zeros((dim,dim,dim,dim))

    # Initialize
    for a, e in product(range(Nelec,dim),range(Nelec,dim)):
        m, i = a-Nelec, e-Nelec
        Fae[a,e] = (1 - (a == e))*fs[a,e]
        Fmi[m,i] = (1 - (m == i))*fs[m,i]
        Fme[m,e] = fs[m,e]

    # Stanton eqs (4), (5), and (6)
    gen_iter = [range(Nelec,dim)]*3
    for a, b, e in product(*gen_iter):
        Fae[a,e] += -0.5*fs[b-Nelec,e]*ts[a,b-Nelec]
        Fmi[a-Nelec,b-Nelec] += 0.5*ts[e,b-Nelec]*fs[a-Nelec,e]
        for f in range(Nelec,dim):
            Fae[a,e] += ts[f,b-Nelec]*spinints[b-Nelec,a,f,e]
            Fmi[a-Nelec,b-Nelec] += ts[e,f-Nelec]*spinints[a-Nelec,f-Nelec,b-Nelec,e] 
            Fme[a-Nelec,e] += ts[f,b-Nelec]*spinints[a-Nelec,b-Nelec,e,f]
            Wmnij[a-Nelec,b-Nelec,e-Nelec,f-Nelec] = spinints[a-Nelec,b-Nelec,e-Nelec,f-Nelec]
            for n in range(Nelec):
                Fae[a,e] += -0.5*taus(a,f,b-Nelec,n)*spinints[b-Nelec,n,e,f]
                Fmi[a-Nelec,b-Nelec] += 0.5*taus(e,n+Nelec,b-Nelec,f-Nelec)*spinints[a-Nelec,f-Nelec,e,n+Nelec]
                Wmnij[a-Nelec,b-Nelec,e-Nelec,f-Nelec] += ts[n+Nelec,f-Nelec]*spinints[a-Nelec,b-Nelec,e-Nelec,n+Nelec] - ts[n+Nelec,e-Nelec]*spinints[a-Nelec,b-Nelec,f-Nelec,n+Nelec]
                for i in range(Nelec):
                    Wmnij[a-Nelec,b-Nelec,e-Nelec,f-Nelec] += 0.25*tau(n+Nelec,i+Nelec,e-Nelec,f-Nelec)*spinints[a-Nelec,b-Nelec,n+Nelec,i+Nelec]

    # Stanton eqs (7) and (8)
    Wabef = np.zeros((dim,dim,dim,dim))
    Wmbej = np.zeros((dim,dim,dim,dim))
    gen_iter.append(range(Nelec,dim))
    for a, b, e, f in product(*gen_iter):
        m, j = a-Nelec, f-Nelec
        Wabef[a,b,e,f] = spinints[a,b,e,f]
        Wmbej[m,b,e,j] = spinints[m,b,e,j]
        for i in range(Nelec):
            Wabef[a,b,e,f] += -ts[b,i]*spinints[a,i,e,f] + ts[a,i]*spinints[b,i,e,f]
            Wmbej[m,b,e,j] += ts[i+Nelec,j]*spinints[m,b,e,i+Nelec]
            Wmbej[m,b,e,j] += -ts[b,i]*spinints[m,i,e,j]
            for n in range(Nelec):
                Wabef[a,b,e,f] += 0.25*tau(a,b,i,n)*spinints[i,n,e,f]
                Wmbej[m,b,e,j] -= (0.5*td[n+Nelec,b,j,i] + ts[n+Nelec,j]*ts[b,i])*spinints[m,i,e,n+Nelec]

    return Fae, Fmi, Fme, Wmnij, Wabef, Wmbej



# makeT1 and makeT2, as they imply, construct the actual amplitudes necessary for computing
# the CCSD energy (or computing an EOM-CCSD Hamiltonian, etc)

# Stanton eq (1)
def makeT1(ts,td):
    tsnew = np.zeros((dim,dim))
    #tsnew = fs
    for a, i in product(range(Nelec,dim),range(Nelec)):
        tsnew[a,i] = fs[i,a]
        for e in range(Nelec,dim):
            tsnew[a,i] += ts[e,i]*Fae[a,e]
        for m in range(Nelec):
            tsnew[a,i] -= ts[a,m]*Fmi[m,i]
            for e in range(Nelec,dim):
                tsnew[a,i] += td[a,e,i,m]*Fme[m,e]
                for f in range(Nelec,dim):
                    tsnew[a,i] -= 0.5*td[e,f,i,m]*spinints[m,a,e,f]
                for n in range(Nelec):
                    tsnew[a,i] -= 0.5*td[a,e,m,n]*spinints[n,m,e,i]
        for n, f in product(range(Nelec),range(Nelec,dim)):
            tsnew[a,i] -= ts[f,n]*spinints[n,a,i,f]
        tsnew[a,i] /= Dai[a,i]
    return tsnew



# Stanton eq (2)
def makeT2(ts,td):
    tdnew = np.zeros((dim,dim,dim,dim))
    gen_iter = [range(Nelec,dim)]*2 + [range(Nelec)]*2

    for a, b, i, j in product(*gen_iter):
        tdnew[a,b,i,j] += spinints[i,j,a,b]
        for e in range(Nelec,dim):
            tdnew[a,b,i,j] += td[a,e,i,j]*Fae[b,e] - td[b,e,i,j]*Fae[a,e]
            for m in range(Nelec):
                tdnew[a,b,i,j] += -0.5*td[a,e,i,j]*ts[b,m]*Fme[m,e] + 0.5*td[b,e,i,j]*ts[a,m]*Fme[m,e]
        for m in range(Nelec):
            tdnew[a,b,i,j] += -td[a,b,i,m]*Fmi[m,j] + td[a,b,j,m]*Fmi[m,i]
            for e in range(Nelec,dim):
                tdnew[a,b,i,j] += -0.5*td[a,b,i,m]*ts[e,j]*Fme[m,e] + 0.5*td[a,b,j,m]*ts[e,i]*Fme[m,e]
        for e in range(Nelec,dim):
            tdnew[a,b,i,j] += ts[e,i]*spinints[a,b,e,j] - ts[e,j]*spinints[a,b,e,i]
            for f in range(Nelec,dim):
                tdnew[a,b,i,j] += 0.5*tau(e,f,i,j)*Wabef[a,b,e,f]
        for m in range(Nelec):
            tdnew[a,b,i,j] += -ts[a,m]*spinints[m,b,i,j] + ts[b,m]*spinints[m,a,i,j]
            for e in range(Nelec,dim):
                tdnew[a,b,i,j] +=  td[a,e,i,m]*Wmbej[m,b,e,j] - ts[e,i]*ts[a,m]*spinints[m,b,e,j]
                tdnew[a,b,i,j] += -td[a,e,j,m]*Wmbej[m,b,e,i] + ts[e,j]*ts[a,m]*spinints[m,b,e,i]
                tdnew[a,b,i,j] += -td[b,e,i,m]*Wmbej[m,a,e,j] + ts[e,i]*ts[b,m]*spinints[m,a,e,j]
                tdnew[a,b,i,j] +=  td[b,e,j,m]*Wmbej[m,a,e,i] - ts[e,j]*ts[b,m]*spinints[m,a,e,i]
            for n in range(0,Nelec):
                tdnew[a,b,i,j] += 0.5*tau(a,b,m,n)*Wmnij[m,n,i,j]
        tdnew[a,b,i,j] /= Dabij[a,b,i,j]
    return tdnew


# Expression from Crawford, Schaefer (2000) 
# DOI: 10.1002/9780470125915.ch2
# Equation (134) and (173)
# computes CCSD energy given T1 and T2
def ccsdenergy():
    ECCSD = 0.0
    gen_iter = [range(Nelec),range(Nelec,dim)]*2
    for i, a, j, b in product(*gen_iter):
         ECCSD += 0.25*spinints[i,j,a,b]*td[a,b,i,j] + 0.5*spinints[i,j,a,b]*(ts[a,i])*(ts[b,j])
    return ECCSD

#================
# MAIN LOOP
# CCSD iteration
#================

ECCSD = 0
DECC = 1.0
while DECC > 0.000000001: # arbitrary convergence criteria
    OLDCC = ECCSD
    Fae,Fmi,Fme,Wmnij,Wabef,Wmbej = updateintermediates()
    tsnew = makeT1(ts,td)
    tdnew = makeT2(ts,td)
    ts = tsnew
    td = tdnew
    ECCSD = ccsdenergy()
    DECC = abs(ECCSD - OLDCC)

print("E(corr,CCSD) = ", ECCSD)
print("E(CCSD) = ", ECCSD + ENUC + EN)
