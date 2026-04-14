import pandas as pd
from tqdm import tqdm
from helper import tprint, scale, pad_temporal_in, y_labels
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


#date elements for dense input
date_meta_dict = {date: [
        date.weekday(),
        date.week,
        np.sin(2 * np.pi * date.weekday() / 7),
        np.cos(2 * np.pi * date.weekday() / 7),
        np.sin(2 * np.pi * date.week / 52),
        np.cos(2 * np.pi * date.week / 52)
    ] for date in obs_dates}

print(f'Gathering observations from {obs_dates[0].date()} to {obs_dates[-1].date()}')

X, dense, Y, stkno_ids = [],[],[],[]

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
    open_trans = trans_df.copy()[trans_df.index.isin(open_orders.index)] # part in open orders
    
    open_trans['trans_date'] = (open_trans['trans_date'] - obs_date).dt.days

    #get metadata about date
    date_meta = date_meta_dict[obs_date]

    for row in open_orders.itertuples(): #group by orders to train as no trans history is valid input
        all_this_trans = open_trans.loc[row.Index] # all transactions
        this_trans = all_this_trans[all_this_trans['trans_date'] < 0].tail(512) # last 512 transactions in the past
        on_hand_qtys = [0,0] if this_trans.empty else list(this_trans[['on_hand', 'wip_on_hand']].iloc[-1]) #most recent stock and wip qtys

        late = row[1]
        min_date_req = row[4]
        qty_req = row[2]
        if late == 0:
            y_label = y_labels['on_time']
        elif min_date_req < 0:
            y_label = y_labels['already_overdue']
        elif on_hand_qtys[0] < qty_req:
            y_label = y_labels['no_stock']
        elif len(all_this_trans[ #if there exists a transaction for this product
                (all_this_trans['trans_date'].between(0, 14)) & # in the proceeding 14 days
                (all_this_trans['correction'] == 1) & # labelled as stock correction
                (all_this_trans['wip'] == 0) & # for finished goods
                (all_this_trans['qty'] < 0) # reducing stock
            ]) > 0:
            y_label = y_labels['stock_wrong']
        else:
            y_label = y_labels['missed_dispatch']

        Y.append(y_label) # 'late' must be called first in groupby
        X.append(this_trans.to_numpy())
        dense.append(
            list(row[2:]) # other elements from open_orders
            + on_hand_qtys # stock and wip levels
            + date_meta # info about date
            + [len(this_trans)] #number of transactions passed to transformer
        )
        stkno_ids.append([row.Index]) # track stck_ids for later splitting


tprint(f'{len(Y)} observations gathered.')

tprint('Padding X')
X, mask = pad_temporal_in(X)

tprint('Scaling data')
X, x_scaler = scale(X, mask = mask)
dense, dense_scaler = scale(dense)
Y = np.stack(Y)
stkno_ids = np.stack(stkno_ids)


obs_dict = {}
tprint('Saving observations to compressed file')
for data, label in zip([X, Y, dense, mask, stkno_ids], ['X', 'Y', 'dense', 'mask', 'stkno_ids']):
    obs_dict[label] = data


np.savez_compressed(rf"cache\all_obs.npz", **obs_dict)

tprint('Saving meta data to pickle file')
with open(r"cache\preprocess_meta.pkl", "wb") as f:
    #scalers as list, temporal input column names
	pickle.dump({'x_scaler': x_scaler, 
                  'dense_scaler': dense_scaler,
                  'columns': open_trans.columns}, f)
     
tprint('Done')