
from sqlalchemy import text, create_engine
import pandas as pd


# rufus_engine = create_engine(r"sqlite:///C:/Python Projects/local.db")

# #create phantom parts view
# with rufus_engine.connect() as conn:
#     print(pd.read_sql(f'''
#             SELECT * FROM acaud WHERE rufus_stkno_id = 630
            
#             ''', con = conn))
# raise
import numpy as np
import pandas as pd

all_obs = np.load(r"cache\all_obs.npz")
X = all_obs['X']
stknos = all_obs['stkno_ids']

for x, stkno in zip(X, stknos):
    if (np.std(x, axis = -1) == 0).any():
        print(pd.DataFrame(data=x)[-60:-30])
        print(stkno)
        raise