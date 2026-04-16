import numpy as np
from helper import tprint, y_labels

tprint('Loading data')
all_obs = np.load(r"cache\tree_all_obs.npz")

tprint('Splitting groups by Y category')
# Generate the data dictionary using comprehension
data = {
    f"{dataset}_{y_group}": all_obs[dataset][all_obs['Y'][:, y_vector.index(1)].astype(bool)] #membership of y category # binary - 0] == y_vector] #
    for y_group, y_vector in y_labels.items() # for each y group
    for dataset in all_obs.keys() # for each dataset
}


tprint('Creating test and train groups by sampling each Y category')
all_ids = np.unique(all_obs['stkno_ids'])
#need a way to generate samples with representation across each y class (and each product group?)
#test_ids = list(np.random.choice(all_ids, size=int(0.2 * len(all_ids)), replace=False)) #reserve 20% of stkno_ids for validation
test_ids = []
# implement greedy algorithm for selecting sample of each y_group


train_idx, test_idx = {}, {}
for y_group in y_labels:
    # stkno_id is in test set filter
    is_test = np.isin(data[f'stkno_ids_{y_group}'].flatten(), list(test_ids))

    #filter for all sets (series <5 makes transformer output nan)
    #all_filter = np.sum(data[f'mask_{y_group}'], axis = 1)>=100 #series larger than 100 samples
    all_filter = True
    train_idx[y_group] = np.random.choice(np.where(~is_test & all_filter)[0], 1300, replace=False)
    #test_idx[y_group] = np.random.choice(np.where(is_test & all_filter)[0], 100, replace=False)


train_data_dict, test_data_dict = {}, {}
for data_label in ['dense', 'Y']: #'X', 'mask'
    train_data_dict[data_label] = np.concatenate([data[f'{data_label}_{y_group}'][train_idx[y_group]] for y_group in y_labels])
    #test_data_dict[data_label] = np.concatenate([data[f'{data_label}_{y_group}'][test_idx[y_group]] for y_group in y_labels])


tprint('Saving data')
np.savez_compressed(r'cache\train.npz', **train_data_dict)
#np.savez_compressed(r'cache\test.npz', **test_data_dict)
