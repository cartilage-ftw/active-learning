import pandas as pd
import numpy as np
import metapredict as meta
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 150
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Serif']
PROTEIN_ALPHABET = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
                     'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


def get_random_aa():
    idx = round(np.floor(np.random.uniform()*20))
    return PROTEIN_ALPHABET[idx]

ubiquitin_seq = 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG'

ub_disorder = meta.predict_disorder_domains(ubiquitin_seq)
print(len(meta.predict_disorder_domains(ubiquitin_seq).disorder))

ub_modif = ubiquitin_seq[:4] + 'AAA' + ubiquitin_seq[7:]
ub_mod_disorder = meta.predict_disorder_domains(ub_modif)

comparison = pd.DataFrame(data={'original': ub_disorder.disorder, 'modified': ub_mod_disorder.disorder})


class Sequence:
    def __init__(self, sequence):
        self.sequence = sequence
    
    def initialize_statistic(self):
        pass



def optimize_sequence(seq: str):
    residue_change_index = round(np.floor(np.random.uniform()*len(seq)))
    modified_seq = seq[:residue_change_index] + get_random_aa() + \
                            seq[residue_change_index:]
    return modified_seq

def handle_optimization(initial_seq: str, steps_before_halt: int = 25):
    # an array that stores the disorder of the sequence on each step
    disorder_steps = []
    seq_steps = []

    disorder_steps.append(sum(meta.predict_disorder(initial_seq)))
    seq_steps.append(initial_seq)

    to_halt = False
    total_attempts = 0
    cumulative_attempts = []
    while (not to_halt):
        current_seq = seq_steps[-1]

        new_seq = optimize_sequence(current_seq)
        new_seq_disorder = sum(meta.predict_disorder(new_seq))

        total_attempts += 1

        if new_seq_disorder > 0.95*disorder_steps[-1]:
            disorder_steps.append(new_seq_disorder)
            seq_steps.append(current_seq)
            cumulative_attempts.append(total_attempts)

        # decide if it should be ended here
        if len(disorder_steps) > steps_before_halt:
            # TODO: what if instead of growing, the disorder is actually declining?
            normalized_growth = np.std(disorder_steps[-steps_before_halt:])/np.average(disorder_steps[-steps_before_halt:])
            print('Normalized growth is', normalized_growth)
            if normalized_growth < 0.05:
                to_halt = True
                print('Halting optimization! Total brute force attempts:', total_attempts)
                print('Cumulative attempts for reaching each optimization step')
    return seq_steps, disorder_steps

def plot_history(sequences: list, disorders: list):
    plt.plot(disorders, marker='o', c='#808080', mfc='#aaaaff', mec='#aaaaff')
    print('*** Sequences probed ****')
    for seq in sequences:
        print(seq)
    plt.xlabel("Optimized Sequence Step")
    plt.ylabel("Residue-wise Disorder Consensus Value Sum")
    plt.tight_layout()
    plt.show()

seqs, disorders = handle_optimization(ubiquitin_seq)
plot_history(seqs, disorders)

'''print(comparison)
print(meta.percent_disorder(ubiquitin_seq))
print(meta.percent_disorder(ub_modif))
print(np.sum(ub_disorder.disorder))
print(np.sum(ub_mod_disorder.disorder))
print(len(ubiquitin_seq))'''
#print(meta.predict_disorder("DSSPEAPAEPPKDVPHDWLYSYVFLTHHPADFLR"))

