import numpy as np

data = np.load(r"cache\observations.npz")
print(data[f'stkno_ids_true'].shape)