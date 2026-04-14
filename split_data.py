data = np.load(r"cache\all_obs.npz")

tprint('Splitting datasets by true/false')
true_obs = (Y.flatten() == 1)
false_obs = (Y.flatten() == 0)
obs_dict = {}

for data, label in zip([X, Y, dense, mask, stkno_ids], ['X', 'Y', 'dense', 'mask', 'stkno_ids']):
    obs_dict[f"{label}_true"] = data[true_obs]
    obs_dict[f"{label}_false"] = data[false_obs]

data = obs_dict

#split data into train and validation sets
all_ids = np.unique(np.concatenate([data['stkno_ids_false'], data['stkno_ids_true']]))
val_ids = list(np.random.choice(all_ids, size=int(0.2 * len(all_ids)), replace=False)) #reserve 20% of stkno_ids for validation


train_idx, val_idx = {}, {}
for suffix in ['true', 'false']:
    val_mask = ~np.isin(data[f'stkno_ids_{suffix}'].flatten(), list(val_ids))
    train_idx[suffix] = np.random.choice(np.where(~val_mask)[0], 20000, replace=True)
    val_idx[suffix] = np.random.choice(np.where(val_mask)[0], 1000, replace=False)

train_data_list, val_data_list = [], []
for data_label in ['X', 'dense', 'mask', 'Y']:
    train_data_list.append(np.concatenate([
            data[f'{data_label}_true'][train_idx['true']],
            data[f'{data_label}_false'][train_idx['false']]
    ]))
    val_data_list.append(np.concatenate([
            data[f'{data_label}_true'][val_idx['true']],
            data[f'{data_label}_false'][val_idx['false']]
    ]))