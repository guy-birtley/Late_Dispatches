import numpy as np

data = np.load(r"cache\all_obs.npz")

Y = data['Y']
print(Y.sum(axis=0))
