
import sys,os,numpy as np
import hoomd, hoomd.md as md
import hoomd.deprecated as old
#from hoomd
import azplugins
import gsd, gsd.hoomd, gsd.pygsd 


##################################################
#################################################
##### 1.1 Parameters  ##############################


#no of polymer1
Npol1=1
#no of polymer3
Npol2=1
#no of polymer2
Npol3=0
            
#amino acid sequence of polymers	
filePol1 = 'sequence.dat'
filePol2 = 'sequence.dat' 
#filePol3 = 'hrd.dat'


#no of monomers in x and y dir for poly 1
Nxpoly1=1
Nzpoly1=1

#no of monomers in x and y dir for poly 2

Nxpoly2=1
Nzpoly2=1

#no of monomers in x and y dir for poly 3

Nxpoly3=0
Nzpoly3=0


# In[2]:

fileroot = 'out_%d'%(Npol1)


#update box size
Lx_update = 9

Ly_update = 9

Lz_update = 9

# SLAB DIMENSIONS
#boxsize=20.0 # The x and y dimensions of the slab configuration in nanometers
#slab_z_length = 80.0 # The z dimension of the slab configuration in nanometers

# RESIZING RUN PARAMETERS
resize_T=400 # Temperature for resizing run in Kelvin
resize_steps=20000 # Number of steps (>500) to be used for resizing the box to boxsize set previously
if resize_steps<500:
    resize_steps=501
resize_dt=0.01 # Time step in picoseconds for box resizing 

# PRODUCTION RUN PARAMETERS
production_dt=0.01 # Time step for production run in picoseconds	
production_steps=1000000000 # Total number of steps

#production_T=int( sys.argv[1]) # Temperature for production run in Kelvin

production_T=300



seq={'R':'ARG','H':'HIS','K':'LYS','D':'ASP','E':'GLU',
     'S':'SER','T':'THR','N':'ASN','Q':'GLN','C':'CYS',
     'U':'SEC','G':'GLY','P':'PRO','A':'ALA','V':'VAL',
     'I':'ILE','L':'LEU','M':'MET','F':'PHE','Y':'TYR',
     'W':'TRP'}


# ##### Read one letter amino acid sequence from file: filePol1 contain sequence of pol1 and filePol2 contain sequence of pol2

### Read one letter amino acid sequence from above files and save each letter in a new file:
Poly1_seq='%s_seq.dat'%(filePol1) 
Poly2_seq='%s_seq.dat'%(filePol2)



nline=1
count=0
k=0
fout=open(Poly1_seq,'w')
with open(filePol1,'r') as fid:
    for i in fid:
        if i[0]!='#':
            for j in i:
                
                if j in seq:
                    k+=1
                    #print(i,j)
                    fout.write(' %s'%seq[j])
                    count+=1
                    if count==nline:
                        fout.write('\n')
                        count=0
fout.close()

NmonPol1=k


# In[5]:




#print('k is',k)
### Read one letter amino acid sequence from above files and save each letter in a new file:

k=0
fout=open(Poly2_seq,'w')
with open(filePol2,'r') as fid:
    for i in fid:
        if i[0]!='#':
            for j in i:
                if j in seq:
                    k+=1
                    fout.write(' %s'%seq[j])
                    count+=1
                    if count==nline:
                        fout.write('\n')
                        count=0
fout.close()
NmonPol2=k


NmonPol3=0


#print(NmonPol1)





#################################################################
# ## 1.2 Read sequence and force field parameters################
# ## Input parameters for all the amino acids (force field)######
# ###############################################################

ff_para = 'new_residues.dat' #'stats_module.dat'
aalist={}
with open(ff_para,'r') as fid:
    for i in fid:
        if i[0]!='#':
            tmp=i.rsplit()
            aalist[tmp[0]]=np.loadtxt(tmp[1:],dtype=float)
aakeys=list(aalist.keys())

#name of all 20 amino acid
# This translates each amino acid type into a number, which will be used in HOOMD
# For example, GLY is with an ID of 10
aamass=[]
aacharge=[]
aaradius=[]
aahps=[]
for i in aakeys:
    aamass.append(aalist[i][0])
    aacharge.append(aalist[i][1])
    aaradius.append(aalist[i][2])
    aahps.append(aalist[i][3])

bond_length=0.38




# Now we can translate the entire sequence of FIRST polymer to a number code according to the order in 'aakeys'
chain_id_pol1 = []
chain_mass_pol1=[]
chain_charge_pol1=[]
with open(Poly1_seq ,'r') as fid:
	for i in fid:
		#print('i is',i)
		iname_pol1=i.rsplit()[0]
		chain_id_pol1.append(aakeys.index(iname_pol1))
		chain_mass_pol1.append(aalist[iname_pol1][0])
		chain_charge_pol1.append(aalist[iname_pol1][1])
		


# Now we can translate the entire sequence of SECOND polymer to a number code according to the order in 'aakeys'
chain_id_pol2 = []
chain_mass_pol2=[]
chain_charge_pol2=[]
with open(Poly2_seq ,'r') as fid:
	for i in fid:
		#print('i is',i)
		iname_pol2=i.rsplit()[0]
		chain_id_pol2.append(aakeys.index(iname_pol2))
		chain_mass_pol2.append(aalist[iname_pol2][0])
		chain_charge_pol2.append(aalist[iname_pol2][1])


len_pol1=len(chain_id_pol1)
len_pol2=len(chain_id_pol2)
len_pol3= 0#len(chain_id_pol3)

box_length=bond_length*(len_pol1+len_pol2+len_pol3)+10

Ntot_pol1=len_pol1*Npol1
Ntot_pol2=len_pol2*Npol2
Ntot_pol3=0#len_pol3*Npol3

Nmon1=NmonPol1

Nmon2=NmonPol2
Nmon3=0#NmonPol3
 

rc=0.4
Dist=0.38




Lx=10+ np.rint ( np.maximum(Lx_update,(np.maximum(Nxpoly1,np.maximum(Nxpoly2,Nxpoly3)))*rc + 2*5*rc))+5


Ly=10+np.rint ( Nmon1*.38 + Nmon2*.38+ Nmon3*.38 ) +5

Lz= 10+np.rint ( np.maximum(Lz_update,(np.maximum(Nzpoly1,np.maximum(Nzpoly2,Nzpoly3)))*rc + 2*5*rc))+5



ycoords1 = np.linspace(-.38*(Nmon1-1)/2 , -.38*(Nmon1-1)/2+.38*(Nmon1-1),Nmon1) # initial z coordinates of brush monomers

ycoords2 = np.linspace(-.38*(Nmon2-1)/2  ,-.38*(Nmon2-1)/2+0.38*(Nmon2-1) ,Nmon2) # initial z coordinates of oligomers

ycoords3 = np.linspace(-.38*(Nmon3-1)/2  ,-.38*(Nmon3-1)/2+0.38*(Nmon3-1) ,Nmon3) # initial z coordinates of oligomers



# ##################################################    
# #### 1.3 Now we can build HOOMD data structure####
# ##################################################



context = hoomd.context.initialize()

snapshot = hoomd.data.make_snapshot(N=Ntot_pol1+Ntot_pol2+Ntot_pol3, 
                                    box=hoomd.data.boxdim(Lx=Lx, Ly=Ly, Lz=Lz,dimensions=3), 
                                    particle_types=aakeys,bond_types=['AA_bond'])

snapshot.bonds.resize((len_pol1-1)*Npol1 + (len_pol2-1)*Npol2 +(len_pol3-1)*Npol3  );




Nx=Nxpoly1+Nxpoly2+Nxpoly3
#Ny=Nypoly1+Nypoly2+Nypoly3
Nz=Nzpoly1+Nzpoly2+Nzpoly3



bz=np.linspace(-Lz/2+3*rc,Lz/2-3*rc,Nz)
bx=np.linspace(-Lx/2+3*rc,Lx/2-3*rc,Nx)


# In[28]:


binz= np.absolute(bz[1]-bz[0])

binx= np.absolute(bx[1]-bx[0])



# In[29]:


#binx=(Lx-2*4*rc)/Nx
#binz=(Lz-2*4*rc)/Nz
#binx




#SNAPSHOT--POSITIONS_ POLY1


tempx = -Lx/2 + 3*rc

for i in range(Nxpoly1):
    tempz = -Lz/2 +  3*rc
    #print(tempx,tempz)

    for j in range(Nzpoly1):
       # print(i,j)
        
        #print(tempx,tempz)
        
        ind = i*Nzpoly1 + j ### index of the brush being initialized
       # print(i,j,ind,ind*Nmon1 , (ind+1)*Nmon1)
        ### position initialization
        #print(tempx,tempz,ycoords1[:])
        snapshot.particles.position[ind*Nmon1:(ind+1)*Nmon1,0] = tempx
        snapshot.particles.position[ind*Nmon1:(ind+1)*Nmon1,2] = tempz
        snapshot.particles.position[ind*Nmon1:(ind+1)*Nmon1,1] = ycoords1[:]
        #print(i,j,tempx,tempz,ycoords1)
        tempz = tempz + binz
        

    tempx = tempx + binx
    
#print(tempx,tempz)
#print(snapshot.particles.position)


#print(tempx,tempz)
#tempz = tempz + binz
#tempx = tempx + binx
#print(tempx,tempz)

for i in range(Nxpoly2):
    tempz = -Lz/2 + 3*rc
    for j in range(Nzpoly2):
        ind = i*Nzpoly2 + j ### index of the brush being initialized
        #print('i,j',i,j,ind,Npol1*Nmon1+ind*Nmon2 , Npol1*Nmon1+(ind+1)*Nmon2)
        ### position initialization
       # print(tempx,tempz,ycoords2[:])
        snapshot.particles.position[Npol1*Nmon1+ind*Nmon2: Npol1*Nmon1+(ind+1)*Nmon2 ,0] = tempx

        snapshot.particles.position[Npol1*Nmon1+ind*Nmon2: Npol1*Nmon1+(ind+1)*Nmon2,2] = tempz
        snapshot.particles.position[Npol1*Nmon1+ind*Nmon2:Npol1*Nmon1+(ind+1)*Nmon2 ,1] = ycoords2[:]
        #print(i,j,tempx,tempz,ycoords2)
        tempz = tempz + binz
        

    tempx = tempx + binx
    
#print(tempx,tempz)
#print(snapshot.particles.position,len(snapshot.particles.position))#






for i in range(Nxpoly3):
    tempz = -Lz/2 + 3*rc
    for j in range(Nzpoly3):
        ind = i*Nzpoly3 + j ### index of the brush being initialized
        #print(i,j,ind,Npol1*Nmon1+Npol2*Nmon2+ind*Nmon3 , Npol1*Nmon1+Npol2*Nmon2+(ind+1)*Nmon3)
        ### position initialization
        #print(tempx,tempz,ycoords3[:])
        snapshot.particles.position[Npol1*Nmon1+Npol2*Nmon2+ind*Nmon3: Npol1*Nmon1+Npol2*Nmon2+(ind+1)*Nmon3 ,0] = tempx

        snapshot.particles.position[Npol1*Nmon1+Npol2*Nmon2+ind*Nmon3: Npol1*Nmon1+Npol2*Nmon2+(ind+1)*Nmon3,2] = tempz
        snapshot.particles.position[Npol1*Nmon1+Npol2*Nmon2+ind*Nmon3: Npol1*Nmon1+Npol2*Nmon2+(ind+1)*Nmon3 ,1] = ycoords3[:]
        #print(tempz)
        #print(tempz)
        tempz = tempz + binz
        

    tempx = tempx + binx
    
#print(tempx,tempz)
#print(snapshot.particles.position,len(snapshot.particles.position))




#SNAPSHOT -- MASS CHARGE  TYPEID

snapshot.particles.typeid[0:Ntot_pol1]=Npol1*chain_id_pol1
snapshot.particles.mass[0:Ntot_pol1] = Npol1 * chain_mass_pol1 
snapshot.particles.charge[0:Ntot_pol1] = Npol1 * chain_charge_pol1

snapshot.particles.typeid[Ntot_pol1:Ntot_pol1+Ntot_pol2] = Npol2*chain_id_pol2
snapshot.particles.mass[Ntot_pol1:Ntot_pol1+Ntot_pol2] = Npol2 * chain_mass_pol2
snapshot.particles.charge[ Ntot_pol1:Ntot_pol1+Ntot_pol2] = Npol2 * chain_charge_pol2



len(snapshot.particles.position)



### bond initialization

nbonds=(len_pol1-1)*Npol1 + (len_pol2-1)*Npol2+(len_pol3-1)*Npol3
bond_pairs=np.zeros((nbonds,2),dtype=int)
snapshot.bonds.resize( nbonds)

snapshot.bonds.typeid[0:nbonds] = [0]*(nbonds)

print('nbonds',nbonds)
k=0
for i in range(Npol1):
    for j in range(NmonPol1):
       # print(i,j)
        if( (i*NmonPol1+j+1) % NmonPol1==0):
            #print('not counted',i,j,i*NmonPol1+j,np.array([i*NmonPol1+j,i*NmonPol1+j+1]))
            k+=1
        else:
            bond_pairs[i*NmonPol1+j-k,:]= np.array([i*NmonPol1+j,i*NmonPol1+j+1])

k=0
n=NmonPol2
m=NmonPol1*Npol1-Npol1
print('m',n,m)
pol2_i=NmonPol1*Npol1
for i in range(Npol2):
    for j in range(NmonPol2):
       # print(i,j)
        if( (i*n+j+1) % n==0):
            #print(m)
            #print('not counted',i,j,i*n+j+pol2_i,np.array([pol2_i+i*n+j,pol2_i+i*n+j+1]))
            k+=1
            #m+=1
        else:
            #print(m)
            bond_pairs[m,:]= np.array([pol2_i +i*n+j,pol2_i+i*n+j+1])
            #print(i,j,np.array([pol2_i +i*n+j,pol2_i+i*n+j+1]))
            m+=1

k=0
n=NmonPol3
m=NmonPol1*Npol1+NmonPol2*Npol2-(Npol1+Npol2)

# In[45]:


pol3_i=NmonPol1*Npol1+NmonPol2*Npol2


pol3_i=NmonPol1*Npol1+NmonPol2*Npol2
for i in range(Npol3):
    for j in range(NmonPol3):
       # print(i*n+j+1)
        #print(m,bond_pairs[m,:])
        if( (i*n+j+1) % n==0):
            #print(m)
           # print('not counted',i,j,i*n+j+pol2_i,np.array([pol3_i +i*n+j,pol3_i+i*n+j+1]))
            k+=1
            #m+=1
        else:
            #print(m)
            bond_pairs[m,:]= np.array([pol3_i +i*n+j,pol3_i+i*n+j+1])
            
            m+=1
            

snapshot.bonds.group[0:nbonds]=bond_pairs[0:nbonds]

system = hoomd.init.read_snapshot(snapshot)

nl = hoomd.md.nlist.cell()
nl.reset_exclusions(exclusions=['1-2', 'body'])

## Bonds
harmonic = hoomd.md.bond.harmonic()
harmonic.bond_coeff.set('AA_bond', k=8360, r0=0.381)
## Nonbonded
nl.reset_exclusions(exclusions=['1-2', 'body'])
nb = azplugins.pair.ashbaugh(r_cut=2.0, nlist=nl)
for i in aakeys:
    for j in aakeys:
        nb.pair_coeff.set(i,j,lam=(aalist[i][3]+aalist[j][3])/2.,
                          epsilon=0.8368, sigma=(aalist[i][2]+aalist[j][2])/10./2.,r_cut=2.0)    

## Electrostatics
yukawa = hoomd.md.pair.yukawa(r_cut=4.0, nlist=nl)
for i,atom1 in enumerate(aakeys):
    for j,atom2 in enumerate(aakeys):
        yukawa.pair_coeff.set(atom1,atom2,epsilon=aalist[atom1][1]*aalist[atom2][1]*1.73136, kappa=1.0, r_cut=4.0) 
        
## Group Particles
all = hoomd.group.all()


xml = old.dump.xml(group=hoomd.group.all(), filename="topology.xml", vis=True)


# #### Set up integrator
dt=0.01
hoomd.md.integrate.mode_standard(dt=dt) # Time units in ps
kTinput=resize_T * 8.3144598/1000.
integrator = hoomd.md.integrate.langevin(group=all, kT=kTinput, seed=63535)


hoomd.update.box_resize(Lx=hoomd.variant.linear_interp([(0,system.box.Lx),(resize_steps-500,Lx_update)]),
                        Ly=hoomd.variant.linear_interp([(0,system.box.Ly),(resize_steps-500,Ly_update)]),
                        Lz=hoomd.variant.linear_interp([(0,system.box.Lz),(resize_steps-500,Lz_update)]),
                        scale_particles=True)
for cnt,i in enumerate(aakeys):
    integrator.set_gamma(i,gamma=aamass[cnt]/1000.0)
    
    
# #### Output log file with box dimensions and restart file after box resizing
#hoomd.analyze.log(filename='resize_tst.log', quantities=['potential_energy','kinetic_energy','temperature','pressure_xx','pressure_yy','pressure_zz','lx','ly','lz'],
 #                                                     period=1000, overwrite=True, header_prefix='#')
hoomd.dump.gsd(fileroot+'resize.gsd', period=200, group=all, truncate=True)

hoomd.run(tsteps=resize_steps)




f = gsd.pygsd.GSDFile(open(fileroot+'resize.gsd','rb'))
t = gsd.hoomd.HOOMDTrajectory(f)

s1 = t[0]


Lx=s1.configuration.box[0]
Ly=s1.configuration.box[1]
Lz=s1.configuration.box[2]
n=s1.particles.N

context_extendbox = hoomd.context.initialize()

s = hoomd.data.make_snapshot(N=n, box=hoomd.data.boxdim(Lx=Lx, Ly=Ly,
    Lz=Lz,xy=0,xz=0,yz=0,dimensions=3), 
    particle_types=s1.particles.types,bond_types=['AA_bond'])




s.bonds.resize(nbonds)

s.bonds.group[0:(len_pol1-1)*Npol1 + (len_pol2-1)*Npol2+(len_pol3-1)*Npol3] = s1.bonds.group[0:(len_pol1-1)*Npol1 + (len_pol2-1)*Npol2 + (len_pol3-1)*Npol3]

s.bonds.typeid[0:nbonds] = [0]*(nbonds)

s.particles.typeid[0:n] = s1.particles.typeid[0:n]

s.particles.mass[0:n] = s1.particles.mass[0:n]
s.particles.charge[0:n] = s1.particles.charge[0:n]
s.particles.position[0:Npol1*NmonPol1] = s1.particles.position[0:Npol1*NmonPol1]
s.particles.position[Npol1*NmonPol1:Npol2*NmonPol2] = s1.particles.position[Npol1*NmonPol1:Npol2*NmonPol2]
s.particles.position[Npol2*NmonPol2:n] = s1.particles.position[Npol2*NmonPol2:n]

system_extend = hoomd.init.read_snapshot(s)

n_steps = production_steps # 1 microseconds

#fileroot = 'testrun_chain_%d_%d_%d'%(Npol1,Npol2,Npol3)
nl = hoomd.md.nlist.cell()

## Bonds
harmonic = hoomd.md.bond.harmonic()
harmonic.bond_coeff.set('AA_bond', k=8360, r0=0.381)
## Nonbonded
nl.reset_exclusions(exclusions=['1-2', 'body'])
nb = azplugins.pair.ashbaugh(r_cut=2.0, nlist=nl)
for i in aakeys:
    for j in aakeys:
        nb.pair_coeff.set(i,j,lam=(aalist[i][3]+aalist[j][3])/2.,
                          epsilon=0.8368, sigma=(aalist[i][2]+aalist[j][2])/10./2.,r_cut=2.0)    

## Electrostatics
yukawa = hoomd.md.pair.yukawa(r_cut=4.0, nlist=nl)
for i,atom1 in enumerate(aakeys):
    for j,atom2 in enumerate(aakeys):
        yukawa.pair_coeff.set(atom1,atom2,epsilon=aalist[atom1][1]*aalist[atom2][1]*1.73136, kappa=1.0, r_cut=4.0) 

## Group Particles
all = hoomd.group.all()

xml = old.dump.xml(group=hoomd.group.all(), filename=fileroot+"topology.xml" , vis=True)


## Set up integrator
hoomd.md.integrate.mode_standard(dt=production_dt) # Time units in ps
temp = production_T*0.00831446
integrator = hoomd.md.integrate.langevin(group=all, kT=temp, seed=399991) # Temp is kT/0.00831446
for cnt,i in enumerate(aakeys):
    integrator.set_gamma(i,gamma=aamass[cnt]/1000.0)
## Outputs
#hoomd.analyze.log(filename=fileroot+'.log', quantities=['potential_energy', 'pressure_xx', 'pressure_yy', 'pressure_zz', 'temperature','lx','ly','lz'], period=10, overwrite=False, header_prefix='#')

hoomd.dump.gsd(fileroot+'restart1.gsd', period=1000000, group=all, truncate=True)
hoomd.dump.dcd(fileroot+'_dump.dcd', period=100000, group=all, overwrite=False)

hoomd.run_upto(production_steps, limit_hours=110)

