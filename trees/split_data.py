import numpy as np
from helper import tprint, y_labels


tprint('Loading data')
all_obs = np.load(r"cache\tree_all_obs.npz")

data_sets = ['dense', 'Y']


tprint('Selecting test and train sets')
all_ids = np.unique(all_obs['stkno_ids'])
#need a way to generate samples with representation across each y class (and each product group?)
#test_ids = list(np.random.choice(all_ids, size=int(0.2 * len(all_ids)), replace=False)) #reserve 20% of stkno_ids for validation
test_ids = []
# implement greedy algorithm for selecting sample of each y_group

test_pool = np.isin(all_obs['stkno_ids'], list(test_ids)) #indicies of test stkno_ids

tprint('Sampling Y groups to create test/train sets')

#initialise dicts
train_data_dict = {k: [] for k in data_sets}
#test_data_dict = {k: [] for k in data_sets}

train_idx, test_idx = {}, {}
for value, text in enumerate(y_labels):
    y_pool = (all_obs['Y'] == value) #indices of Y group

    train_idx = np.random.choice(np.where(y_pool & ~test_pool)[0], 1500, replace=False)  # in y group and in train group
    # test_idx = np.random.choice(np.where(y_pool & test_pool)[0], 200, replace=False)  # in y group and in test group


    for data_set in data_sets:
        train_data_dict[data_set].append(all_obs[data_set][train_idx])
        #test_data_dict[data_set].append(all_obs[data_set][test_idx])

#concetenate to numpy arrays
for k in train_data_dict:
    train_data_dict[k] = np.concatenate(train_data_dict[k], axis=0)


tprint('Saving data')
np.savez_compressed(r'cache\tree_train.npz', **train_data_dict)
#np.savez_compressed(r'cache\test.npz', **test_data_dict)
