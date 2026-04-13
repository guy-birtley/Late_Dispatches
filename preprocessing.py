import pandas as pd
from tqdm import tqdm
from helper import tprint, scale, pad_temporal_in
import numpy as np
import pickle

##### static parameters #####

forecast_horizon = 2
obs_dates = pd.date_range(pd.Timestamp(2025, 1, 1), pd.Timestamp(2025, 12, 20), freq='B')

tprint('Reading data and preprocessing')

# read data from db_queries script
orders_df = pd.read_pickle(r"cache\orders_df.pkl")
trans_df = pd.read_pickle(r"cache\trans_df.pkl")

#convert to datetime objects
trans_df['trans_date'] = pd.to_datetime(trans_df['trans_date'])
order_date_cols = ['req_date','order_date','desp_date']
orders_df[order_date_cols] = orders_df[order_date_cols].apply(pd.to_datetime)

# get forecast window (forecast horizon business days before due)
orders_df['forecast_window'] = orders_df['req_date'] - pd.offsets.BusinessDay(n=forecast_horizon)

#forward fill stock on hand figures
for col in ['on_hand', 'wip_on_hand']:
    trans_df[col] = trans_df.groupby(level=0)[col].ffill().fillna(0)



#date elements for list index
date_meta_dict = {date: [date.weekday(), date.week] for date in obs_dates}

print(f'Gathering observations from {obs_dates[0]} to {obs_dates[-1]}')

X, dense, Y = [],[],[]

for obs_date in tqdm(obs_dates):

    #get open orders as of midnight on obs_date
    open_orders = orders_df.copy()[(orders_df['forecast_window'] <= obs_date) & #due in less than forecast_horizon working days
                            (orders_df['desp_date'] >= obs_date) # despatched after observation
                        ].groupby(level = 0).agg({
                            'late':'max', # if any open order for that product shipped late
                            'qty':['sum','count'], #total qty and total orders
                            'req_date':['min', 'max'] #first and last due date of order
                        })
    #put min/max due dates as integers relative to obs_date
    open_orders['req_date'] = (open_orders['req_date'] - obs_date).apply(lambda x: x.dt.days)
    open_orders.columns = ['_'.join(col) for col in open_orders.columns.values] #multi to single columns
    
    #get transactions of open order parts before observation date
    open_trans = trans_df.copy()[(trans_df.index.isin(open_orders.index)) # part in open orders
                                 & (trans_df['trans_date'] < obs_date) # transactions occured before obs date
                                ].groupby(level=0).tail(512) #limit to 512 (max for MOMENT transformer)
    open_trans['trans_date'] = (open_trans['trans_date'] - obs_date).dt.days

    #get metadata about date
    date_meta = date_meta_dict[obs_date]

    for row in open_orders.itertuples(): #group by orders to train as no trans history is valid input
        this_prod_open_trans = open_trans.loc[row.Index]
        on_hand_qtys = [0,0] if this_prod_open_trans.empty else list(this_prod_open_trans[['on_hand', 'wip_on_hand']].iloc[-1])
        X.append(this_prod_open_trans.to_numpy())
        dense.append(
            list(row[2:]) # other elements from open_orders
            + on_hand_qtys # stock and wip levels
            + date_meta # info about date
            + [len(this_prod_open_trans)] #number of transactions passed to transformer
        )
        Y.append([row[1]]) # 'late' must be called first in groupby


tprint(len(Y), 'observations gathered.')

tprint('Scaling data')
X, x_scaler = scale(X)
dense, dense_scaler = scale(dense)
Y = np.stack(Y)

tprint('Padding X')
X, mask = pad_temporal_in(X)

tprint('Splitting datasets by true/false')
true_obs = (Y.flatten() == 1)
false_obs = (Y.flatten() == 0)
obs_dict = {}
for data, label in zip([X, Y, dense, mask], ['X', 'Y', 'dense', 'mask']):
    obs_dict[f"{label}_true"] = data[true_obs]
    obs_dict[f"{label}_false"] = data[false_obs]


tprint('Saving observations to compressed file')
np.savez_compressed(rf"cache\observations.npz", **obs_dict)


tprint('Saving meta data to pickle file')
with open(r"cache\preprocess_meta.pkl", "wb") as f:
    #scalers as list, temporal input column names
	pickle.dump({'x_scaler': x_scaler, 
                  'dense_scaler': dense_scaler,
                  'columns': open_trans.columns}, f)
     
tprint('Done')