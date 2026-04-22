import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


all_obs = np.load(r"cache\all_obs.npz")

dense, Y = all_obs['dense'], all_obs['Y']

print(all_obs['dense'].shape, all_obs['Y'].shape)

rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rf.fit(dense, Y)

importances = rf.feature_importances_

print(importances)

result = permutation_importance(rf, dense, Y, n_repeats=10, random_state=42)

perm_importances = result.importances_mean

print(perm_importances)