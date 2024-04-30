import sys
import localcider
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from localcider.sequenceParameters import SequenceParameters


plt.rcParams['figure.dpi'] = 150
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Serif']

slice_after = 50
args = sys.argv
if len(args) > 1:
		slice_after = int(args[1])

seqs_init = pd.read_csv("../data/seq_b22_data_init.csv").iloc[:slice_after]
'''seqs_init50_r1 = pd.read_csv("../data/new/init50/seq_b22_data_50_r1.csv")
seqs_init50_r2 = pd.read_csv("../data/new/init50/seq_b22_data_50_r2.csv")
seqs_init50_r3 = pd.read_csv("../data/new/init50/seq_b22_data_init50_r3.csv")'''

init50_r1_seqs = pd.read_csv('../data/new/init50/seq_b22_data_50_r1.csv')
init50_r2_seqs = pd.read_csv('../data/new/init50/seq_b22_data_50_r2.csv')
init50_r3_seqs = pd.read_csv('../data/new/init50/seq_b22_data_init50_r3.csv')
init50_r4_seqs = pd.read_csv('../data/new/init50/seq_b22_opt50_r4.csv')
init50_r5_seqs = pd.read_csv('../data/new/init50/seq_b22_data_init50_r5.csv')

init100_r1_seqs = pd.read_csv('../data/new/init100/seq_b22_init100_r1.csv')
init150_r1_seqs = pd.read_csv('../data/new/init150/seq_init150_r1.csv')
init200_r1_seqs = pd.read_csv('../data/new/init200/seq_b22_data_init200_r1.csv')

init100_r2_seqs = pd.read_csv('../data/new/init100/seq_b22_init100_r2.csv')
init150_r2_seqs = pd.read_csv('../data/new/init150/seq_b22_init150_r2.csv')
init200_r2_seqs = pd.read_csv('../data/new/init200/seq_b22_init200_r2.csv')

#seqs_init100_r1 = pd.read_csv("../data/new/init100/seq_b22_init100_r1.csv")
#seqs_init100_r2 = pd.read_csv('')


def estimate_descriptors(df):
	"""
	Arguments:
		df: A (pandas) DataFrame object, which contains a column containing sequences; it's fine to pass
		your dataframe which additionally has both sequences and B22, this method just appends the descriptors
		in separate columns in the same dataframe

	Returns:
		A new dataframe with additional columns containing each of the descriptors
	"""

	# define a dictionary that will hold the descriptor values for each seq
	descriptor_dict = {
		'Fraction Charged Residues': [], # initialize to be empty
		'Net Charge Per Residue': [],
		'Fraction Negatively Charged': [],
		'Fraction Positively Charged': [],
		'Fraction Expanding': [], # residues that contribute to expansion of the chain
		'Disorder Promoting Residues': [],
		'Kappa': [],
		'Omega': [],
		'Mean Hydropathy': []
	} 

	for i in range(len(df)):
		seq = df.iloc[i]['Sequence']
		# create an object localCIDER uses
		seq_obj = SequenceParameters(seq)
		descriptor_dict['Fraction Charged Residues'].append(seq_obj.get_FCR())
		descriptor_dict['Net Charge Per Residue'].append(seq_obj.get_NCPR(pH=None))
		descriptor_dict['Fraction Negatively Charged'].append(seq_obj.get_fraction_negative())
		descriptor_dict['Fraction Positively Charged'].append(seq_obj.get_fraction_positive())
		descriptor_dict['Fraction Expanding'].append(seq_obj.get_fraction_expanding(pH=None))
		descriptor_dict['Disorder Promoting Residues'].append(seq_obj.get_fraction_disorder_promoting())
		descriptor_dict['Kappa'].append(seq_obj.get_kappa())
		descriptor_dict['Omega'].append(seq_obj.get_Omega())
		descriptor_dict['Mean Hydropathy'].append(seq_obj.get_mean_hydropathy())

	descriptor_df = pd.DataFrame(data=descriptor_dict)
	combined_df = pd.concat([df, descriptor_df], axis=1)
	
	return combined_df


def make_violin_plots(dataframe_list):
	"""
	Instead of a scatter plot, turn those into violin plots
	"""
	fig, axes = plt.subplots(3,3, figsize=(9,9))
	# the name of the column in the dataframes, corresponding to the descriptors
	# of interest
	descriptor_col_names = ['Fraction Charged Residues', 'Net Charge Per Residue',
						 'Fraction Negatively Charged', 'Fraction Positively Charged',
						 'Fraction Expanding', 'Disorder Promoting Residues', 
						 'Kappa', 'Omega', 'Mean Hydropathy']
	descriptor_arr_dict = {}
	# for violin plots, we need a single array containing all iters
	# start by combining df's
	iterations = np.arange(len(dataframe_list))
	for i, iteration_data in enumerate(dataframe_list):
		for desc in descriptor_col_names:
			if desc in descriptor_arr_dict:
				descriptor_arr_dict[desc].append(iteration_data[desc].to_numpy())
			else:
				descriptor_arr_dict[desc] = [iteration_data[desc].to_numpy()]
	
	cmap = matplotlib.colormaps['Spectral_r'] # Spectral_r also looked fine
	colors = cmap(np.arange(0, 1, 1/len(iterations)), bytes=True)/255

	# now, for each descriptor, place the data in one of the panels as a violin plot
	for desc, (i, j) in zip(descriptor_col_names, np.ndindex(3,3)):
		axes[i,j].set_prop_cycle(color=colors)
		for it in iterations:
			parts = axes[i, j].violinplot(descriptor_arr_dict[desc][it], [it], showmedians=True,
								showextrema=True, bw_method=0.2)
			axes[i,j].scatter(it, np.median(descriptor_arr_dict[desc][it]), c=colors[it], marker='+', s=128)
			# customize plot
			for pc in parts['bodies']:
				#pc.set_facecolor(colors[it])
				pc.set_edgecolor('k')
				#pc.set_alpha(1)

		axes[i, j].set_xlabel('Iteration', fontsize=13)
		axes[i, j].set_xticks(iterations)
		axes[i, j].set_ylabel(desc, fontsize=13)
		axes[i, j].tick_params(which='major', axis='both', length=6, labelsize=13)
	plt.tight_layout()
	plt.savefig("descriptors_violin.png", dpi=300)


def make_plots(dataframe_list):
	# haha lol
	fig, axes = plt.subplots(3, 3, figsize=(9,9))
	for i, seq_set in enumerate(dataframe_list):
		axes[0,0].scatter(i*np.ones(len(seq_set)), seq_set['Fraction Charged Residues'], s=8, alpha=0.6,
					 c=seq_set['Numeric Label'], marker='$\circ$')
		axes[0,0].set_ylabel('Fraction Charged Residues')

		axes[0,1].scatter(i*np.ones(len(seq_set)), seq_set['Net Charge Per Residue'], s=8, alpha=0.6,
					c=seq_set['Numeric Label'], marker='$\circ$')
		axes[0,1].set_ylabel('Net Charge Per Residue')

		axes[0,2].scatter(i*np.ones(len(seq_set)), seq_set['Fraction Negatively Charged'], s=8, alpha=0.6,
					c=seq_set['Numeric Label'], marker='$\circ$')
		axes[0,2].set_ylabel('Fraction Negatively Charged')

		axes[1,0].scatter(i*np.ones(len(seq_set)), seq_set['Fraction Positively Charged'], s=8, alpha=0.6,
					c=seq_set['Numeric Label'], marker='$\circ$')
		axes[1,0].set_ylabel('Fraction Positively Charged')

		axes[1,1].scatter(i*np.ones(len(seq_set)), seq_set['Fraction Expanding'], s=8, alpha=0.6,
					c=seq_set['Numeric Label'], marker='$\circ$')
		axes[1,1].set_ylabel('Fraction Expanding')

		axes[1,2].scatter(i*np.ones(len(seq_set)), seq_set['Disorder Promoting Residues'], s=8, alpha=0.6,
					c=seq_set['Numeric Label'], marker='$\circ$')
		axes[1,2].set_ylabel('Disorder Promoting Residues')
		
		axes[2,0].scatter(i*np.ones(len(seq_set)), seq_set['Kappa'], s=8, alpha=0.6,
					c=seq_set['Numeric Label'], marker='$\circ$')
		axes[2,0].set_ylabel('Kappa')

		axes[2,1].scatter(i*np.ones(len(seq_set)), seq_set['Omega'], s=8, alpha=0.6,
					c=seq_set['Numeric Label'], marker='$\circ$')
		axes[2,1].set_ylabel('Omega')

		axes[2,2].scatter(i*np.ones(len(seq_set)), seq_set['Mean Hydropathy'], s=8, alpha=0.6,
					c=seq_set['Numeric Label'], marker='$\circ$')
		axes[2,2].set_ylabel('Mean Hydropathy')

		#cax = plt.axes((0.85, 0.1, 0.075, 0.8))
		#fig.colorbar(axes[2,2], cax=cax)
	for i,j in np.ndindex((3,3)):
		axes[i,j].set_xlabel('Iteration')
	plt.tight_layout()
	plt.savefig('descriptors_init50.png', dpi=300)
	plt.show()


def compare_b22_distrib(dataframe_list):
	fig, ax = plt.subplots(figsize=(6,6))

	cmap = matplotlib.colormaps['RdYlBu']
	colors = cmap(np.arange(0, 1, 1/len(dataframe_list)), bytes=True)/255

	for i, df in enumerate(dataframe_list):
		if 'Numeric Label' in df.columns:
			ax.hist(-df['Numeric Label'], bins=12, alpha = 0.4, fc=colors[i], ec='k', lw=0.25, label=f'Iteration {i}')
			ax.axvline(x=-np.average(df['Numeric Label']), ymin=0, ymax=1, ls='--', c=colors[i], lw=0.75)
			print(f'Mean B22 in Iter {i}:', np.average(df['Numeric Label']))
	ax.legend()
	ax.set_xlabel('$-B_{22}$', fontsize=13)
	ax.set_ylabel('Frequency [per bin]', fontsize=13)
	plt.savefig('B22_distrib_iter.png', dpi=150)


def plot_amino_acid_fractions(dataframe_list):
	
	averaged_aa_fracs = {}
	num_iter = len(dataframe_list)

	fig, ax = plt.subplots(figsize=(9,6))
	for i, data in enumerate(dataframe_list):
		# a dictionary to hold amino acid fractions for all sequences
		# in this iteration
		amino_acid_dict = {}
		
		#print(data.head())
		for seq in data['Sequence']:
			seq_obj = SequenceParameters(seq)
			aa_fracs = seq_obj.get_amino_acid_fractions()
			for amino_acid, fraction in aa_fracs.items():
				if amino_acid in amino_acid_dict.keys():
					amino_acid_dict[amino_acid].append(fraction)
				else:
					amino_acid_dict[amino_acid] = [fraction]
		# if the dictionary hasn't been initialized, do it now
		if len(averaged_aa_fracs) == 0:
			for amino_acid in amino_acid_dict.keys():
				averaged_aa_fracs[amino_acid] = []
		# create a new dataframe, with the sequences dataframe appended with amino acid fractions
		data_with_fracs = pd.concat([data,
							pd.DataFrame(data=amino_acid_dict)], axis=1)
		# now I want to take an average along columns; first remove any columns containing
		# strings, so that we only have floats
		data_without_seq = data_with_fracs.drop(['Sequence', ], axis=1)
		averaged_data = data_without_seq.mean(axis=0) # column-wise average
		# now append these averaged fractions to the dictionary of AAs
		for amino_acid in averaged_aa_fracs.keys():
			averaged_aa_fracs[amino_acid].append(averaged_data[amino_acid])

	# I later realized in the data in a different form, so I'll modify again
	amino_acid_letters = []
	amino_acid_fracs_list = []
	for aa, frac_list in averaged_aa_fracs.items():
		amino_acid_letters.append(aa)
		amino_acid_fracs_list.append(frac_list)
	# take transpose of the array 
	amino_acid_frac_arr = np.array(amino_acid_fracs_list).T

	bar_width = 0.25
	bar_spacing = bar_width*num_iter + 0.5
	#x_positions = np.arange(0, num_iter*bar_width, bar_width)
	x_pos = np.arange(0., 20.*bar_spacing, bar_spacing)
	# choose discrete colors from a color map
	cmap = matplotlib.colormaps['YlGn_r']
	colors = cmap(np.arange(0, 1, 1/num_iter), bytes=True)/255
	# bytes=True gets you RGBA colors between 0-255, I divided by 255 to normalize

	for iteration in range(num_iter):
		ax.bar(x_pos, amino_acid_frac_arr[iteration], bar_width, label=f'Iteration {iteration}',
		 			fc=colors[iteration], ec='k', lw=0.2)
		x_pos += bar_width

	ax.legend(fontsize=14)
	ax.set_xticks(x_pos - (num_iter/2)*bar_width, amino_acid_letters)
	ax.tick_params(axis='both', which='major', length=6, labelsize=14)
	ax.set_xlabel('Amino Acid', fontsize=14)
	ax.set_ylabel('Fraction of Sequence', fontsize=14)

	ax.text(x=0, y=0.55, s=r'$n_{\textrm{init}}$' + f'$ = {slice_after}$', fontsize=14)
	plt.tight_layout()
	plt.savefig(f'amino_acid_fractions_init{slice_after}.png', dpi=300)
	plt.show()



if __name__ == "__main__":
	dataframe_list = [seqs_init, init50_r1_seqs, init50_r2_seqs,
							  init50_r3_seqs, init50_r4_seqs, init50_r5_seqs]#[]
	df_descriptors = []
	for i, seq_set in enumerate(dataframe_list):#([seqs_init, init50_r1_seqs, init50_r2_seqs,
							  #init50_r3_seqs, init50_r4_seqs, init50_r5_seqs]):#seqs_init50_r1, seqs_init50_r2, seqs_init50_r3]):
		seq_set['Iteration'] = i*np.ones(len(seq_set))
		df_descriptors.append(estimate_descriptors(seq_set))
	#make_plots(df_descriptors)
	make_violin_plots(df_descriptors)
	compare_b22_distrib(dataframe_list)
	#plot_amino_acid_fractions(dataframe_list)
	plt.show()