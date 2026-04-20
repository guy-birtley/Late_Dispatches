import numpy as np
import pandas as pd
from helper import tprint, y_labels

'''
This creates equal distribution of each class for both test and train sets.
Instead, I should take a k fold split for the test set, then take equal sample of each group for non-test sets.
'''


tprint('Loading data')
all_obs = np.load(r"cache\all_obs.npz")

data_sets = ['mask','X', 'dense', 'Y']


tprint('Selecting test and train sets')

train_sample_size = 700
test_sample_size = 50

df = pd.DataFrame(index = all_obs['stkno_ids'], data = {'y': all_obs['Y']})

unique_ids = np.unique(all_obs['stkno_ids'])
np.random.shuffle(unique_ids)

test_ids, train_ids = set(), set()

test_counts, train_counts = np.zeros(len(y_labels)), np.zeros(len(y_labels))

for id in unique_ids:
    y_counts = df.loc[[id]].groupby('y').size()
    y_counts_list = np.array([y_counts.get(i, 0) for i in range(len(y_labels))])

    if any((train_counts < train_sample_size)*(y_counts_list > 0)): # stkno_id in incomplete training set
        train_ids.add(id)
        train_counts = np.add(train_counts, y_counts_list)
    elif (any(test_counts < test_sample_size)): # stkno_id in test set
        test_ids.add(id)
        test_counts = np.add(test_counts, y_counts_list)
    else:
        break

if any(train_counts < train_sample_size) or any(test_counts < test_sample_size):
    tprint('Split_failed')
    print(train_counts)
    print(test_counts)
    raise


test_pool = np.isin(all_obs['stkno_ids'], list(test_ids)) #indicies of test stkno_ids
train_pool = np.isin(all_obs['stkno_ids'], list(train_ids))

tprint('Sampling Y groups to create test/train sets')

#initialise dicts
train_data_dict = {k: [] for k in data_sets}
test_data_dict = {k: [] for k in data_sets}

train_idx, test_idx = {}, {}
for value, text in enumerate(y_labels):
    y_pool = (all_obs['Y'] == value) #indices of Y group

    train_idx = np.random.choice(np.where(y_pool & train_pool)[0], train_sample_size, replace=False)  # in y group and in train group
    test_idx = np.random.choice(np.where(y_pool & test_pool)[0], test_sample_size, replace=False)  # in y group and in test group


    for data_set in data_sets:
        train_data_dict[data_set].append(all_obs[data_set][train_idx])
        test_data_dict[data_set].append(all_obs[data_set][test_idx])

#concetenate to numpy arrays
for k in train_data_dict:
    train_data_dict[k] = np.concatenate(train_data_dict[k], axis=0) #classes in order - must be trained in shuffled batches
    test_data_dict[k] = np.concatenate(test_data_dict[k], axis=0)



tprint('Saving data')

np.savez_compressed(r'cache\train.npz', **train_data_dict)
np.savez_compressed(r'cache\test.npz', **test_data_dict)
