import pandas as pd
import numpy as np
import wazy
import jax
import matplotlib.pyplot as plt
import time

import warnings
# I wanted to suppress the stupid warnings in the terminal output
warnings.filterwarnings("default", category=UserWarning)
# but this didn't get rid of them -_-

plt.rcParams['figure.dpi'] = 150
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Serif']

key = jax.random.PRNGKey(0)

"""
Load the sequences to train with
"""

seq_set1 = pd.read_csv('../data/seq_b22.csv')
seq_set2 = pd.read_csv('../data/data_set2.csv')
seq_set3 = pd.read_csv('../data/data_set3_1.csv')
seq_set4 = pd.read_csv('../data/seq_b22_data4.csv')

# total set:
seq_b22 = pd.concat([seq_set1, seq_set2, seq_set3, seq_set4])
# validation set
validation_seqs = pd.read_csv('../data/validation_set.csv')


def train_bo_seqs(bo_alg, training_data, test_data, predict_labels=True):
    """
    This method does a pure learning of the Bayesian Optimizer
    and records a "training history", at each training step by storing the 
    residual between the prediction and test set
    """
    # store how the prediction performs on the validation data
    num_training_steps = len(training_data)
    training_hist = pd.DataFrame(data={'Sequence':test_data['Sequence']})

    for i in range(len(training_data)):
        sequence, computed_b22_val = training_data.iloc[i]
        bo_alg.tell(key, sequence, computed_b22_val)

        if predict_labels == True:
            predicted_labels = []
            # now check how well the prediction goes
            print(f'Predicting seq labels after training step {i+1}')
            ti = time.time()
            for j in range(len(test_data)):
                test_seq, test_seq_b22, seq_source_category = test_data.iloc[j]
                pred_b22 = bo_alg.predict(key, test_seq)
                # pred_b22 has shape (3,) and contains prediction, err, and epistemic err
                predicted_labels.append(pred_b22)
            tf = time.time()
            print(f'Took {tf-ti} secs to predict for {len(predicted_labels)} sequences in')
            # add predictions of this step to history
            training_hist[f'Pred {i+1} Residual'] = np.array(predicted_labels)[:, 0] - test_data['Numeric Label']
            # for uncertainty, sum both statistical and systematic uncertainty in quadrature
            training_hist[f'Pred {i+1} Err'] = np.sqrt(np.array(predicted_labels)[:, 1]**2
                                            + np.array(predicted_labels)[:, 2]**2)
    print('Successfully trained with', len(training_data), ' sequences')
    
    return num_training_steps, training_hist


def plot_training_hist(training_hist, num_train_steps, train_lengths=None):
    """
    Plots the mean squared deviation every training step
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
    print("predicted error", avg_pred_err)
    plt.plot(steps, mse, label=r'ProtGPT2 Validation Set')
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
    plt.xlabel('Training Step', fontsize=13)
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
        plt.xlabel('Training Step', fontsize=13)
        plt.ylabel('$B_{22}$ Residual', fontsize=13)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    b_optimizer = wazy.BOAlgorithm()
    num_train_steps, training_hist = train_bo_seqs(b_optimizer, seq_b22, validation_seqs, predict_labels=False)
    
    #num_train_steps2, training_hist2 = train_bo_seqs(b_optimizer, seq_set4, validation_seqs, predict_labels=True)
    #date_stamp = time.strftime("%b%d", time.gmtime()).lower() # e.g. feb14
    #training_hist2.to_csv(f"training_hist_{date_stamp}.csv", index=False)
    #plot_training_hist(training_hist2, num_train_steps2)

    total_train_steps = 42 + 21
    '''training_hist = pd.read_csv('training_hist_feb10.csv')
    train_hist2 = pd.read_csv('training_hist_feb11_last.csv')
    train_hist2 = train_hist2.drop('Sequence', axis=1)'''
    total_train_hist = pd.read_csv('training_hist_feb11_prot_gpt.csv')#pd.concat([training_hist, train_hist2], axis=1)
    #total_train_hist.to_csv('training_hist_feb11.csv', index=False)
    #num_train_steps = 42
    plot_training_hist(total_train_hist, total_train_steps, [21, 10, 11, 21])
    #plot_training_indiv(training_hist, num_train_steps)

    print('Next sequences to simulate:', b_optimizer.batch_ask(key, 21))
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
