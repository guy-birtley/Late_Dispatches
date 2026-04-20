import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import pickle
from helper import tprint


if __name__ == "__main__": # for multiple spawns

    tprint('Loading data')
    train_data_raw = np.load(r'cache\train.npz')

    tprint('Fitting tree')
    # Average CV score on the training set was: 0.7165384615384615
    GBTree = GradientBoostingClassifier(
        learning_rate=0.5, #how much each tree contributes
        max_depth=10, #depth of each tree
        max_features=0.35, #fraction of features seen by each tree
        min_samples_leaf=7, #minimum size of leaf resulting from a split
        min_samples_split=14, #minimum size to attempt a split (saves attempted splits guarenteed to fail min_sample_leaf)
        n_estimators=100, #number of trees
        subsample=1.0 #fraction of data used per tree (use all rows)
    )

    GBTree.fit(train_data_raw['dense'],train_data_raw['Y'])

    tprint('Saving model')
    with open(r"cache\model_gbtree.pkl", "wb") as f:
        pickle.dump(GBTree, f)

    tprint('Loading test data')
    test_data_raw = np.load(r'cache\test.npz')

    tprint('Running predictions')
    Y_hat = GBTree.predict(test_data_raw['dense'])

    print('Accuracy', (Y_hat == test_data_raw['Y']).mean())

    