import pandas as pd
import numpy as np
import wazy, jax
import matplotlib.pyplot as plt
import time, sys

from sklearn.model_selection import train_test_split

import warnings
# I wanted to suppress the stupid warnings in the terminal output
warnings.filterwarnings("default", category=UserWarning)
# but this didn't get rid of them -_-


plt.rcParams['figure.dpi'] = 150
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Serif']

# seed to use for the pseudo random generator
seed = 0
key = jax.random.PRNGKey(seed)

"""
Load the sequences to train with
"""


seq_set1_new = pd.read_csv('../data/seq_b22_data_init.csv')
seq_set1 = pd.read_csv('../data/seq_b22.csv')
'''
seq_set2 = pd.read_csv('../data/data_set2.csv')
seq_set3 = pd.read_csv('../data/data_set3_1.csv')
seq_set4 = pd.read_csv('../data/seq_b22_data4.csv')
seq_set5 = pd.read_csv('../data/seq_b22_data_5.csv')'''

# total set:
seq_b22 = pd.concat([seq_set1_new, seq_set1]) #, seq_set2, seq_set3, seq_set4, seq_set5
# validation set
validation_seqs = pd.read_csv('../data/validation_set.csv')
# NOTE: There are about 22 sequences in the validation set from the An et al. paper
# those have very negative B_22 values. For the present purpose, I'm discarding those
validation_seqs = validation_seqs.iloc[42:].reset_index(drop=True)

print(validation_seqs.head())

def train_bo_seqs(bo_alg, training_data, test_data, predict_labels=True, optimize_for='negative', start_count=0):
	"""
	This method does a pure learning of the Bayesian Optimizer
	and records a "training history", at each training step by storing the 
	residual between the prediction and test set
	"""
	# store how the prediction performs on the validation data
	num_training_steps = len(training_data)
	# tracks the history as we go along
	indiv_step_hist = []

	training_data.reset_index(drop=True, inplace=True)
	test_data.reset_index(drop=True, inplace=True)

	print("Initiating optimization")
	for i in range(len(training_data)):
		sequence, computed_b22_val = training_data.iloc[i]
		if optimize_for == 'negative':
			bo_alg.tell(key, sequence, -computed_b22_val)
		else:
			bo_alg.tell(key, sequence, computed_b22_val)
		if predict_labels == True:
			predicted_labels = []
			# now check how well the prediction goes
			print(f'Predicting seq labels after training step {start_count+i+1}')
			ti = time.time()
			for j in range(len(test_data)):
				test_seq, test_seq_b22, seq_source_category = test_data.iloc[j]
				pred_b22 = bo_alg.predict(key, test_seq)
				# pred_b22 has shape (3,) and contains prediction, err, and epistemic err
				if optimize_for == 'negative':
					predicted_labels.append(-np.array(pred_b22))
				else:
					predicted_labels.append(np.array(pred_b22))
			tf = time.time()
			print(f'Took {tf-ti} secs to predict for {len(predicted_labels)} sequences in step {start_count+i+1}')
			# add predictions of this step to history
			if optimize_for == 'negative':
				predictions = pd.Series(-np.array(predicted_labels)[:, 0], name=f'Pred {start_count+i+1} B22')
			else:
				predictions = pd.Series(np.array(predicted_labels)[:, 0], name=f'Pred {start_count+i+1} B22')	
			residuals = pd.Series(np.array(predicted_labels)[:, 0] - test_data['Numeric Label'].to_numpy(),
						 	name=f'Pred {start_count + i + 1} Residual')
			# for uncertainty, sum both statistical and systematic uncertainty in quadrature
			errors = pd.Series(np.sqrt(np.array(predicted_labels)[:, 1]**2
											+ np.array(predicted_labels)[:, 2]**2),
							 name=f'Pred {i+start_count+1} Err')
			#training_hist = pd.concat([training_hist, residuals, errors], axis=1)
			indiv_step_hist.append(predictions)
			indiv_step_hist.append(residuals)
			indiv_step_hist.append(errors)
	# sequences against which the training was evaluated against
	tested_sequences = test_data['Sequence']
	training_hist = pd.concat([tested_sequences, *indiv_step_hist], axis=1)
	print('Successfully trained with', len(training_data), ' sequences')
	
	return num_training_steps, training_hist




def plot_training_hist_legacy(training_hist, num_train_steps, train_lengths=None):
	"""
	Plots the mean squared deviation every training step

	NOTE: This is not the method in use now; when I started working, I was storing residuals and 
	uncertainties in the prediction. I've switched to storing the actual predicted values
	"""
	steps = np.linspace(1, num_train_steps, num_train_steps)
	mse = []
	avg_pred_err = []

	fig, ax = plt.subplots()
	for i in range(num_train_steps):
		mse.append(np.average(np.array(training_hist[f'Pred {i+1} Residual'])**2))
		avg_pred_err.append(np.average(training_hist[f'Pred {i+1} Err']))

	mse = np.array(mse)
	avg_pred_err = np.array(avg_pred_err) # to allow the use of +/- operation on entire list
	#print("predicted error", avg_pred_err)
	plt.plot(steps, mse, label=r'Validation Set')
	plt.fill_between(steps, y1=mse+avg_pred_err, y2=mse-avg_pred_err, alpha=0.6,)


	# highlight/shade the different epochs in different colors
	if train_lengths != None:
		last_step = 0
		odd_counter = 0
		for i in range(len(train_lengths)):
			ax.fill_betweenx(y=[0, np.max(mse)], x1=last_step, x2=last_step + train_lengths[i],
							  fc='steelblue', alpha=0.2+0.2*odd_counter)
			last_step += train_lengths[i]
			odd_counter = (odd_counter + 1)%2

	ax.legend(loc='upper right')
	ax.tick_params(axis='both', which='major', length=6, labelsize=13)
	ax.axhline(y=0, xmin=0, xmax=1, ls='--', lw=0.5, c='dimgray')
	plt.xlabel('Optimization Step', fontsize=13)
	plt.ylabel('Mean Squared Error', fontsize=13)
	
	plt.tight_layout()

	date_stamp = time.strftime("%b%d", time.gmtime()).lower() # e.g. feb14
	plt.savefig(f'training_summary_{date_stamp}.png', dpi=300)
	plt.show()


def predict_for_sequences(bo_algorithm, sequence_list, save_file_name='predictions.csv'):
	"""
	A utility method for when I want to see what the 
	"""
	predictions = []
	for seq  in sequence_list:
		predictions.append(bo_algorithm.predict(key, seq))
	pred_array = np.array(predictions)
	pred_dataframe = pd.DataFrame(data={
				'Sequence': sequence_list,
				'Prediction': pred_array[:,0],
				'Prediction Uncertainity': np.sqrt(pred_array[:,1]**2 + pred_array[:,2]**2)
				})
	pred_dataframe.to_csv(save_file_name, index=False)
	

def plot_training_indiv(training_hist, num_train_steps):
	steps = np.linspace(1, num_train_steps, num_train_steps)
	for seq, seq_hist in training_hist.groupby('Sequence'):
		fig, ax = plt.subplots()
		preds = []
		errors = []
		for i in range(num_train_steps):
			preds.append(seq_hist[f'Pred {i+1} Residual'].iloc[0])
			errors.append(seq_hist[f'Pred {i+1} Err'].iloc[0])
		plt.errorbar(steps, preds, yerr=errors, marker='o', c='#ff557f', ls='', capsize=3, label='$B_{22}$ Residual')
		#plt.fill_between(steps, y1=mse+avg_pred_err, y2=mse-avg_pred_err, alpha=0.6,)
		plt.suptitle('Sequence: '+ seq)
		ax.tick_params(axis='both', which='major', length=6, labelsize=13)
		ax.axhline(y=0, xmin=0, xmax=1, ls='--', lw=0.5, c='dimgray')
		plt.xlabel('Optimization Step', fontsize=13)
		plt.ylabel('$B_{22}$ Residual', fontsize=13)
		plt.tight_layout()
		plt.show()


if __name__ == "__main__":

	next_seqs = ['EAKSAHSNKPEKRRKQQPREF', 'MEEAPARRRRQRPKQAPQREL', 'RKKEREEMSEPPKSQSQEQRE',
   	'RPPRKKPAAARAPSQPSKKRR', 'TQAKREPRRPEPPRRRRPQSE', 'MQQQRPRKRSEPPARHRRRSQ',
	'ETAQARPRQPRPPRKQKQRQK', 'QTESHEEPPQKKPRPAKRSRR', 'KRRQSKEQPAARRQKKQATKQ',
	'QAAHRRPNRRQSQQKPATKRR', 'QRQSREEWPARPPAQPPRRRR', 'MEREESEQRPAEPKQPRRRRR',
 	'KSQKQQRRRRAPPQARRAKPA', 'CTKRASPRRAKAEAPPQDKEC', 'MEESQQPRQARQPRSRRPPQE',
  	'RAKSRQPREPKKRRRQQSKQQ', 'ETRSDHQHQPQPPRKPRRKMR', 'EQEQEAPAPPKQRKSQRQSQQ',
	'REPLDSPTPRQPKHKSRRFSE', 'MSPESSKVRPKQEQSPPRQQT', 'RTRSRATRARKPPRKKQRRAE',
 	'KTRAPRRRPPRTQEETKEQQR', 'MRQQQEQQATEEEQQRPRPPP', 'MAESQQPKRPRRSSARQEPPQ', 'KEMSTHKRRPRPTQARRQEPA']
	# The keyword arguments passed to python when running this file
	args = sys.argv

	init50_r1_seqs = pd.read_csv('../data/new/init50/seq_b22_data_50_r1.csv')
	init50_r2_seqs = pd.read_csv('../data/new/init50/seq_b22_data_50_r2.csv')
	init50_r3_seqs = pd.read_csv('../data/new/init50/seq_b22_data_init50_r3.csv')
	# Initialize the Bayesian Optimization algorithm
	b_optimizer = wazy.BOAlgorithm()
	slice_after = 50
	
	init_seq_ctd = pd.read_csv('../data/ctd/sequences_and_labels_0.csv')
	if len(args) > 1:
		slice_after = int(args[1])

	ctd_train_data, ctd_test_data = train_test_split(init_seq_ctd, test_size=20, shuffle=True)
	ctd_test_data['Source Category'] = ['wherever']*len(ctd_test_data)
	num_train_init, train_hist_init = train_bo_seqs(b_optimizer, ctd_train_data, ctd_test_data, #seq_b22.iloc[:slice_after], validation_seqs,
												 optimize_for='positive', predict_labels=True)
	#predict_for_sequences(b_optimizer, next_seqs)

	'''num_train_step_r1, training_hist_r1 = train_bo_seqs(b_optimizer, init50_r1_seqs, validation_seqs,
												 predict_labels=False, start_count=slice_after)
	num_train_steps_r2, training_hist_r2 = train_bo_seqs(b_optimizer, init50_r2_seqs, validation_seqs,
												 predict_labels=False, start_count=slice_after+25)
	num_train_steps_r3, training_hist_r3 = train_bo_seqs(b_optimizer, init50_r2_seqs, validation_seqs,
												 predict_labels=True, start_count=slice_after+50)'''
	'''num_train_steps, training_hist = train_bo_seqs(b_optimizer, seq_b22.iloc[slice_after:slice_after+50], validation_seqs,
												 predict_labels=True, start_count=slice_after)'''

	'''time_stamp = time.strftime("%b%d_%H-%M", time.gmtime()).lower() # e.g. feb14
	training_hist_r3.to_csv(f"training_hist_{time_stamp})an_et_al.csv", index=False)
	'''
	#print('Next sequences to simulate:', b_optimizer.batch_ask(key, 25))
	#training_hist.
	'''training_hist_prev = pd.read_csv('training_hist_feb27_subset1.csv')
	train_hist_50_100 = pd.read_csv("training_hist_51_100.csv")
	train_hist_100_150 = pd.read_csv('training_hist_mar20_101_150.csv')
	train_hist_150_200 = pd.read_csv('training_hist_mar20_151_200.csv')'''
	'''total_train_hist = pd.concat([training_hist_prev, train_hist_50_100.drop('Sequence', axis=1),
							   train_hist_100_150.drop('Sequence', axis=1),
							   train_hist_150_200.drop('Sequence', axis=1)],
							   axis=1)'''
	
	#next25_seq_hist = pd.read_csv('training_hist_mar22_12-08.csv')
	'''train_hist_init = pd.read_csv('training_hist_init50_r2_an_etal.csv')
	cumulative_hist = pd.concat([train_hist_init, training_hist_r3.drop('Sequence', axis=1)]#training_hist_r1.drop("Sequence", axis=1),
							  #training_hist_r2.drop('Sequence', axis=1)]
							 , axis=1)
	cumulative_hist.to_csv('training_hist_init50_r3_an_etal.csv', index=False)
	#training_hist.to_csv(f"training_hist_{time_stamp}.csv", index=False)
	#total_train_hist.to_csv(f"total_training_hist_mar20.csv", index=False)
	#plot_training_hist(training_hist_prev, 50, [50])
	
	plot_training_hist_legacy(cumulative_hist, 125, [50, 25, 25, 25])'''
	train_hist_init.to_csv('ctd_condensate_train50.csv', index=False)
	plot_training_hist_legacy(train_hist_init, num_train_init, [50])