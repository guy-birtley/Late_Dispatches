import numpy as np
from helper import tprint, y_labels


tprint('Loading data')
all_obs = np.load(r"cache\all_obs.npz")

data = {}
for y_group, y_vector in y_labels.items(): #for each y category
    y_members = all_obs['Y'][:, y_vector.index(1)].astype(bool) #membership of category mask
    for dataset in ['X', 'Y', 'dense', 'mask', 'stkno_ids']: #for each data set
        data[f"{dataset}_{y_group}"] = all_obs[dataset][y_members] # where category is 1]


#split data into train and validation sets
all_ids = np.unique(all_obs['stkno_ids'])
test_ids = list(np.random.choice(all_ids, size=int(0.2 * len(all_ids)), replace=False)) #reserve 20% of stkno_ids for validation


train_idx, test_idx = {}, {}
for y_group in y_labels:
    is_test = np.isin(data[f'stkno_ids_{y_group}'].flatten(), list(test_ids))
    train_idx[y_group] = np.random.choice(np.where(~is_test)[0], 1000, replace=True)
    test_idx[y_group] = np.random.choice(np.where(is_test)[0], 100, replace=False)

train_data_dict, test_data_dict = {}, {}
for data_label in ['X', 'dense', 'mask', 'Y']:
    train_data_dict[data_label] = np.concatenate([data[f'{data_label}_{y_group}'][train_idx[y_group]] for y_group in y_labels])
    test_data_dict[data_label] = np.concatenate([data[f'{data_label}_{y_group}'][test_idx[y_group]] for y_group in y_labels])

np.savez_compressed(r'cache\train.npz', **train_data_dict)
np.savez_compressed(r'cache\test.npz', **test_data_dict)
