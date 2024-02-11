import pandas as pd

root_dir = '../data/'


data5 = pd.read_csv(root_dir + 'data_set5.csv', names=['Sequence', 'B22'])
data4 = pd.read_csv(root_dir + 'data_set4.csv', names=['Sequence', 'B22'])
data6 = pd.read_csv(root_dir + 'data_set6.csv', names=['Sequence', 'B22'])
seq_list_fine = pd.read_csv(root_dir + 'seqs/after_finetuning.txt', names=['Sequence'])
seq_list_orig = pd.read_csv(root_dir + 'seqs/protgpt2.txt', names=['Sequence'])


fine_tuned_seq = []
fine_tuned_b22 = []
orig_seq = []
orig_b22 = []

for i in range(1, len(data4)):
    seq, b22 = data4.iloc[i]
    if seq in list(seq_list_fine['Sequence']):
        fine_tuned_seq.append(seq)
        fine_tuned_b22.append(b22)
    else:
        orig_seq.append(seq)
        orig_b22.append(b22)
    seq, b22 = data5.iloc[i]
    if seq in list(seq_list_fine['Sequence']):
        fine_tuned_seq.append(seq)
        fine_tuned_b22.append(b22)
    else:
        orig_seq.append(seq)
        orig_b22.append(b22)


orig_df = pd.DataFrame(data={'Sequence': orig_seq, 'B22': orig_b22, 'Source':
                             ['ProtGPT Standard']*len(orig_seq)})
fine_tuned_df = pd.DataFrame(data={'Sequence': fine_tuned_seq, 'B22': fine_tuned_b22,
                        'Source': ['ProtGPT Vasilis']*len(fine_tuned_seq)})

# data 6, which has sequences taken from An et al. can be directly used as is
data6['Source'] = ['An et al. (2024)']*len(data6)

all_validation = pd.concat([orig_df, fine_tuned_df, data6])
#all_validation.to_csv('../data/validation_set.csv', index=False)

# there are some duplicate sequeunces, I wanted to spot which ones those are.
# apparently, I realized the duplicate seq has same AA seq but diff calcualted B22
all_b22 = list(all_validation['Sequence'])
print([b22 if (all_b22.count(b22) > 1) else 0 for b22 in all_b22])
print(orig_df.head())