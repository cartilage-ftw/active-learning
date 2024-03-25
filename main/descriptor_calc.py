import localcider
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from localcider.sequenceParameters import SequenceParameters


plt.rcParams['figure.dpi'] = 150
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Serif']


seqs_init = pd.read_csv("../data/seq_b22_data_init.csv").iloc[:50]
seqs_init50_r1 = pd.read_csv("../data/new/init50/seq_b22_data_50_r1.csv")
seqs_init50_r2 = pd.read_csv("../data/new/init50/seq_b22_data_50_r2.csv")
seqs_init50_r3 = pd.read_csv("../data/new/init50/seq_init50_r3.csv")

seqs_init100_r1 = pd.read_csv("../data/new/init100/seq_b22_init100_r1.csv")
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


def make_plots(dataframe_list):
	# haha lol
	fig, axes = plt.subplots(3, 3, figsize=(9,9))
	for i, seq_set in enumerate(dataframe_list):
		axes[0,0].scatter(i*np.ones(len(seq_set)), seq_set['Fraction Charged Residues'], s=8, alpha=0.6)
		axes[0,0].set_ylabel('Fraction Charged Residues')

		axes[0,1].scatter(i*np.ones(len(seq_set)), seq_set['Net Charge Per Residue'], s=8, alpha=0.6)
		axes[0,1].set_ylabel('Net Charge Per Residue')

		axes[0,2].scatter(i*np.ones(len(seq_set)), seq_set['Fraction Negatively Charged'], s=8, alpha=0.6)
		axes[0,2].set_ylabel('Fraction Negatively Charged')

		axes[1,0].scatter(i*np.ones(len(seq_set)), seq_set['Fraction Positively Charged'], s=8, alpha=0.6)
		axes[1,0].set_ylabel('Fraction Positively Charged')

		axes[1,1].scatter(i*np.ones(len(seq_set)), seq_set['Fraction Expanding'], s=8, alpha=0.6)
		axes[1,1].set_ylabel('Fraction Expanding')

		axes[1,2].scatter(i*np.ones(len(seq_set)), seq_set['Disorder Promoting Residues'], s=8, alpha=0.6)
		axes[1,2].set_ylabel('Disorder Promoting Residues')
		
		axes[2,0].scatter(i*np.ones(len(seq_set)), seq_set['Kappa'], s=8, alpha=0.6)
		axes[2,0].set_ylabel('Kappa')

		axes[2,1].scatter(i*np.ones(len(seq_set)), seq_set['Omega'], s=8, alpha=0.6)
		axes[2,1].set_ylabel('Omega')

		axes[2,2].scatter(i*np.ones(len(seq_set)), seq_set['Mean Hydropathy'], s=8, alpha=0.6)
		axes[2,2].set_ylabel('Mean Hydropathy')

	for i,j in np.ndindex((3,3)):
		axes[i,j].set_xlabel('Iteration')
	plt.tight_layout()
	plt.savefig('descriptors_init50.png', dpi=300)
	plt.show()


def compare_b22_distrib(dataframe_list):
	fig, ax = plt.subplots(figsize=(6,6))
	for i, df in enumerate(dataframe_list):
		if 'Numeric Label' in df.columns:
			ax.hist(-df['Numeric Label'], bins='fd', alpha = 0.4, ec='k', lw=0.25, label=f'Iteration {i}')
			ax.axvline(x=-np.average(df['Numeric Label']), ymin=0, ymax=1, ls='--', c='gray', lw=0.75)
	ax.legend()
	ax.set_xlabel('$-B_{22}$', fontsize=13)
	ax.set_ylabel('Frequency [per bin]', fontsize=13)
	plt.savefig('B22_distrib_iter.png', dpi=150)

if __name__ == "__main__":
	dataframe_list = []
	for i, seq_set in enumerate([seqs_init, seqs_init50_r1, seqs_init50_r2, seqs_init50_r3]):
		seq_set['Iteration'] = i*np.ones(len(seq_set))
		dataframe_list.append(estimate_descriptors(seq_set))
		
	#make_plots(dataframe_list)
	compare_b22_distrib(dataframe_list)
	plt.show()