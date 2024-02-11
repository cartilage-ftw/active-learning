import freud
import gsd.hoomd
import matplotlib.pyplot as plt
import numpy as np
import sys
from itertools import *
import MDAnalysis.units  # for bulk water density
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import sys
import glob


def correctRDF(r,rdf,L):
    dr = r[1]-r[0]
    V = L**3
    Nouter = 1 - 4/3*np.pi*r**3 / V
    DeltaN = 1/V*4*np.pi*dr*np.cumsum( (rdf - 1 ) * r**2)
    return rdf * Nouter / ( Nouter  - DeltaN)
    # Ganguly and van der Vegt, DOI: 10.1021/ct301017q


def calcB22(r,rdf,MW):
    B22 = -2*np.pi*np.trapz((rdf-1)*r*r,r)
    return B22,B22* 6.022e23 / 1e21 / MW**2 * 1e3 # µl mol / g2


def kd_from_b22(b22,L,MW):
    # https://pubs.acs.org/doi/10.1021/acs.jpcb.9b11802
    # Avogadro's number in units 1/mol
    N_A = 6.02214076*1e23
    # Volume of simulation box in L
    V = ((L*1e-10)**3)*1000
    # conversion factor for b22 (from µl mol / g2 to L) 
    convert_units = (MW**2)*1e-6/N_A
    return 1/(N_A*b22*convert_units)


traj_list = glob.glob('../data/trajectories*/results*/frame*.xtc')
u = mda.Universe('FIDO/CLONE90/frame0.tpr', traj_list[::10])

# Initialize freud RDF object
nbins = 30
rmax = 29
rdf_bincount = np.zeros(nbins)
bincount=np.zeros((0,nbins))

rdf = freud.density.RDF(bins=nbins, r_max=rmax, r_min=0.38)

k=0
rdf_err=[]
for i in filename[:]:
    
    #print(i)
    
    reader = mda.coordinates.XTC.XTCReader(i)
    for frame in reader[::500]:
        k+=1

        points1 = frame.positions[:291] # chain1
        points2 = frame.positions[291:582] #chain2
        
        box = freud.box.Box.from_matrix(frame.triclinic_dimensions)
        
        rdf.compute(system=(box, points1), query_points=points2,  reset=False)
        
        if k % 500 == 0:
            rdf_err.append(rdf.rdf)
            
            #print(k)
            
np.savetxt('rdf.txt',rdf.rdf)

g=rdf.rdf

MW_aa = u.residues[:21].atoms.masses
L=u.dimensions[0]
crdf = correctRDF(r,g,L)
b22,B22= calcB22(r,crdf,MW_aa.sum())

print('B22',B22)
