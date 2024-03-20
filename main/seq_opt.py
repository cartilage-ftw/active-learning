import pandas as pd
import numpy as np
import wazy, jax
import matplotlib.pyplot as plt
import time, sys

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
validation_seqs = validation_seqs.iloc[:42]


def train_bo_seqs(bo_alg, training_data, test_data, predict_labels=True, start_count=0):
	"""
	This method does a pure learning of the Bayesian Optimizer
	and records a "training history", at each training step by storing the 
	residual between the prediction and test set
	"""
	# store how the prediction performs on the validation data
	num_training_steps = len(training_data)
	# tracks the history as we go along
	indiv_step_hist = []

	print("Initiating optimization")
	for i in range(len(training_data)):
		sequence, computed_b22_val = training_data.iloc[i]
		bo_alg.tell(key, sequence, -computed_b22_val)
		if predict_labels == True:
			predicted_labels = []
			# now check how well the prediction goes
			print(f'Predicting seq labels after training step {start_count+i+1}')
			ti = time.time()
			for j in range(len(test_data)):
				test_seq, test_seq_b22, seq_source_category = test_data.iloc[j]
				pred_b22 = bo_alg.predict(key, test_seq)
				# pred_b22 has shape (3,) and contains prediction, err, and epistemic err
				predicted_labels.append(-np.array(pred_b22))
			tf = time.time()
			print(f'Took {tf-ti} secs to predict for {len(predicted_labels)} sequences in step {start_count+i+1}')
			# add predictions of this step to history
			predictions = pd.Series(-np.array(predicted_labels)[:, 0], name=f'Pred {start_count+i+1} B22')
			residuals = pd.Series(np.array(predicted_labels)[:, 0] - test_data['Numeric Label'],
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
	# The keyword arguments passed to python when running this file
	args = sys.argv

	# Initialize the Bayesian Optimization algorithm
	b_optimizer = wazy.BOAlgorithm()
	slice_after = 50
	
	if len(args) > 1:
		slice_after = int(args[1])

	num_train_init, train_hist_init = train_bo_seqs(b_optimizer, seq_b22.iloc[:slice_after], validation_seqs,
												 predict_labels=False)

	num_train_steps, training_hist = train_bo_seqs(b_optimizer, seq_b22.iloc[slice_after:slice_after+50], validation_seqs,
												 predict_labels=True, start_count=slice_after)

	print('Next sequences to simulate:', b_optimizer.batch_ask(key, 25))
	time_stamp = time.strftime("%b%d_%H-%M", time.gmtime()).lower() # e.g. feb14
	training_hist.to_csv(f"training_hist_{time_stamp}.csv", index=False)

	#training_hist.
	training_hist_prev = pd.read_csv('training_hist_feb27_subset1.csv')
	train_hist_50_100 = pd.read_csv("training_hist_51_100.csv")
	train_hist_100_150 = pd.read_csv('training_hist_mar20_101_150.csv')
	train_hist_150_200 = pd.read_csv('training_hist_mar20_151_200.csv')
	total_train_hist = pd.concat([training_hist_prev, train_hist_50_100.drop('Sequence', axis=1),
							   train_hist_100_150.drop('Sequence', axis=1),
							   train_hist_150_200.drop('Sequence', axis=1)],
							   axis=1)
	
	#training_hist.to_csv(f"training_hist_{time_stamp}.csv", index=False)
	#total_train_hist.to_csv(f"total_training_hist_mar20.csv", index=False)
	#plot_training_hist(training_hist_prev, 50, [50])
	#plot_training_hist_legacy(total_train_hist, 200, [200])
	'''num_train_steps2, training_hist2 = train_bo_seqs(b_optimizer, seq_set5, validation_seqs, predict_labels=True)
	
	plot_training_hist(training_hist2, num_train_steps2)'''

	#total_train_steps = 63 + 21
	#training_hist = pd.read_csv('training_hist_feb11 (copy).csv')
	#train_hist2 = pd.read_csv('training_hist_feb14_last.csv')
	#train_hist2 = xverl
	#total_train_hist = pd.read_csv('training_hist_feb14.csv')#pd.concat([training_hist, train_hist2], axis=1)
	#otal_train_hist.to_csv('training_hist_feb14.csv', index=False)
	#num_train_steps = 42
	#plot_training_hist(total_train_hist, total_train_steps, [21, 10, 11, 21, 21])
	#plot_training_indiv(training_hist, num_train_steps)

	   # print('Using validation sequences for training')
	'''for i in range(len(validation_seqs)):
		seq, label = validation_seqs.iloc[i]
		b_optimizer.tell(key, seq, label)
		print('Successfully trained with ', seq)'''

#seq_to_predict = 'YSPTSPSYSPTSPSYSPTSPS'
#print('Predicted value for sequence', seq_to_predict)
#print(b_optimizer.predict(key, seq_to_predict))

'''print('Asking Wazy to suggest next 10 sequences')
ti = time.time()
print('Next 10 sequences to predict:\n', b_optimizer.batch_ask(key, 10))
tf = time.time()
delta_t = tf - ti
print('Time taken for this prediction', delta_t)'''


'''data = pd.read_csv('b22_bo_training.csv')

for index, step_data in data.groupby('Index'):
	fig, ax = plt.subplots(figsize=(6,6))
	print('Number of steps for this sequence', len(step_data))
	plt.errorbar(x=np.linspace(21-len(step_data), 21, len(step_data)),
				 y=step_data['Prediction'],
				 yerr=step_data['Pred_uncert'], capsize=3, ls='', marker='o', c='#aaaaff', label=f'Seq {index}')
	plt.axhline(y=seq_b22['Numeric Label'].iloc[index], xmin=0, xmax=1, c='tab:pink', label='True Computed Values')
	plt.xlabel('Training Step')
	plt.ylabel('Predicted $B_{2}$')
	plt.title('Sequence: ' + seq_b22['Sequence'].iloc[index])
	plt.tight_layout()
	plt.legend()
	plt.savefig('pred_figures/seq_{0}_pred.png'.format(index), dpi=150)
	#plt.show()'''
