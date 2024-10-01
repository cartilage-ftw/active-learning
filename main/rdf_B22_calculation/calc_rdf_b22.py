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

import pandas as pd
# Initialize freud RDF object

path=sys.argv[2]
sequence_file=sys.argv[1]


def calculate_standard_error(data_column):

    if len(data_column) == 0:
        raise ValueError("Input array 'data_column' is empty.")

    error_den = np.std(data_column) / np.sqrt(len(data_column))
    return error_den
def adjust_rdf_for_density(radii, rdf_values, box_length):
    """
    Adjust the radial distribution function (RDF) for density effects.

    Parameters:
    radii (numpy.ndarray): Array of radial distances.
    rdf_values (numpy.ndarray): Original RDF values.
    box_length (float): Length of the cubic simulation box.

    Returns:
    numpy.ndarray: Adjusted RDF values.
    """
    dr = radii[1] - radii[0]  # Calculate the radial step size
    volume = box_length ** 3  # Calculate the volume of the simulation box
    N_outer = 1 - (4/3) * np.pi * radii**3 / volume  # Calculate the outer number density
    delta_n = (1 / volume) * 4 * np.pi * dr * np.cumsum((rdf_values - 1) * radii**2)  # Adjust for density

    # Return the adjusted RDF
    return rdf_values * N_outer / (N_outer - delta_n)

def compute_b22(radii, rdf_values, molecular_weight):
    """
    Calculate the second virial coefficient (B22) from the RDF.

    Parameters:
    radii (numpy.ndarray): Array of radial distances.
    rdf_values (numpy.ndarray): Adjusted RDF values.
    molecular_weight (float): Molecular weight of the substance.

    Returns:
    float: The second virial coefficient (B22) in µl mol / g².
    """
    b22 = -2 * np.pi * np.trapz((rdf_values - 1) * radii * radii, radii)  # Compute B22 using trapezoidal integration

    return b22 * 6.022e23 / 1e21 / molecular_weight**2 * 1e3 # convert to unit

print(len(sequences))        
for i in range(1,len(sequences)+1):
    seqNo=i
    print(path+'/../seq{}/out_1_dump.dcd'.format(seqNo))
    print(i)
    reader = mda.coordinates.DCD.DCDReader(path+'/../seq{}/out_1_dump.dcd'.format(seqNo))

    L=reader.dimensions[0]
    nbins = 30
    rmax = 3.8 #*L/8
    k=0
    rdf_bincount = np.zeros(nbins)
    bincount=np.zeros((0,nbins))
    rdf = freud.density.RDF(bins=nbins, r_max=rmax, r_min=0.38)
    rdf_array=np.zeros((reader.n_frames,30))
    len_seq=int(reader.n_atoms/2) #len(sequences)
                
    print((len_seq))
    for frame in reader:

        

        points1 = frame.positions[:len_seq] #u.select_atoms('resid 0:21') #snap.particles.position[:polyCount1] 
        points2 = frame.positions[len_seq:] #snap.particles.position[14000:14000+polyCount2] #freud.locality.AABBQuery(box, pos2).points

        #nlist = aq.query( points1, {"r_max": r_max, "exclude_ii": True}).toNeighborList()

        #rdf.compute(system=(box, points1), query_points=points2,  reset=False)

        box = freud.box.Box.from_matrix(frame.triclinic_dimensions)

        rdf.compute(system=(box, points1), query_points=points2)
        #rdf.compute(system=(box, points1), query_points=points2,  reset=False)
        rdf_array[k] = rdf.rdf
        k+=1

        #if k % 500 == 0:
            #rdf_err.append(rdf.rdf)

            #print(k)
    file_path = path+"rdf_seq{}.txt".format(seqNo)

    # Write the array to a text file
    np.savetxt(file_path, rdf_array)
   
    
# calculate B22


B22_array=np.zeros(len(sequences))

for i in range(1,1+len(sequences)):

    # Load rdf data from a file
    rdf_seq1 = np.genfromtxt(path+'rdf_seq{}.txt'.format(i))

    # Calculate the standard error of the mean for each column in rdf_seq1
    rdf_error_list = [calculate_standard_error(rdf_seq1[:, j]) for j in range(30)]
    rdf_error_list = np.array(rdf_error_list)

    # Calculate mean values for each column in rdf_seq1
    mean_rdf = rdf_seq1.mean(axis=0)

    r=rdf.bin_centers
    u=mda.Universe(path+'../seq{}/topology.xml'.format(10),path+'../seq{}/out_1_dump.dcd'.format(10))
    L=u.dimensions[0]

    natoms=int(u.atoms.n_atoms/2)
    MW = u.atoms[:natoms].masses #u.residues[:21].atoms.masses

    crdf = correctRDF(r,mean_rdf,L)
    B22= calcB22(r,crdf,MW.sum())
    B22_array[i-1]=B22



numeric_label = B22_array

df = pd.DataFrame({'Sequence': sequences, 'Numeric Label': numeric_label})

file_name = path+'/data'

df.to_csv(file_name + '.csv', index=False)
