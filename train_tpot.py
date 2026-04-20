from helper import tprint
import numpy as np
import joblib
from tpot import TPOTClassifier
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="stopit")


if __name__ == "__main__": # for multiple spawns

    tprint('Loading data')
    train_data_raw = np.load(r'cache\train.npz')

    Y = train_data_raw['Y']
    dense = train_data_raw['dense']

    print(dense.shape)
    print(Y.shape)
    
    tpot = TPOTClassifier(
        generations=100,
        population_size=50,
        verbosity=2,
        n_jobs=-1,
        max_time_mins=120,        # 1–2 hours runtime
        periodic_checkpoint_folder="tpot_checkpoints"
    )

    tpot.fit(dense, Y)

    tpot.export(r"cache\best_pipeline.py")
    joblib.dump(tpot.fitted_pipeline_, r"cache\best_model.pkl")
    