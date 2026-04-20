import pandas as pd
from tqdm import tqdm 
import numpy as np
import pickle
from helper import tprint, y_labels, pad_temporal_in, scale

##### static parameters #####

forecast_horizon = 2
obs_dates = pd.date_range(pd.Timestamp(2025, 1, 1), pd.Timestamp(2025, 12, 20), freq='B')
max_placeholder = np.iinfo(np.int32).max

tprint('Reading data and preprocessing')


# read data from db_queries script
orders_df = pd.read_pickle(r"cache\orders_df.pkl")
trans_df = pd.read_pickle(r"cache\trans_df.pkl")
stck_df = pd.read_pickle(r"cache\stck_df.pkl")

#convert to datetime objects
trans_df['trans_date'] = pd.to_datetime(trans_df['trans_date'])
order_date_cols = ['req_date','order_date','desp_date']
orders_df[order_date_cols] = orders_df[order_date_cols].apply(pd.to_datetime)

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
        np.cos(2 * np.pi * date.week / 52),
        (date + pd.offsets.BusinessDay(forecast_horizon) - date).days # days in forecast horizon
    ] for date in obs_dates}

#convert stck table to list dictionary
stck_dict = stck_df.T.to_dict('list')

stocktake_dict = trans_df[trans_df['wip'] == 0].groupby(level=0)['on_hand'].first().to_dict() #first stock value of the year by stkno

tprint(f'Gathering observations from {obs_dates[0].date()} to {obs_dates[-1].date()}')


X, dense, Y, stkno_ids = [],[],[],[]

for obs_date in tqdm(obs_dates):
    #get metadata about date
    date_meta = date_meta_dict[obs_date]

    forecast_days = date_meta[-1]

    forecast_horizon_date = obs_date + pd.offsets.BusinessDay(n=forecast_horizon)
    #get open orders as of midnight on obs_date
    open_orders = orders_df.copy()[(orders_df['req_date'] <= forecast_horizon_date) & #due in less than forecast_horizon working days
                            (orders_df['desp_date'] >= obs_date) # despatched after observation
                        ].groupby(level = 0).agg({
                            # 'late':'max', # if any open order for that product shipped late
                            'desp_date': 'max',
                            'qty':['sum','count'], #total qty and total orders
                            'req_date':['min', 'max'] #first and last due date of order
                        })
    #put min/max due dates as integers relative to obs_date
    open_orders['req_date'] = (open_orders['req_date'] - obs_date).apply(lambda x: x.dt.days)
    open_orders.columns = ['_'.join(col) for col in open_orders.columns.values] #multi to single columns
    open_orders['desp_date_max'] = (open_orders['desp_date_max'] > forecast_horizon_date).astype(int) #shipped after forecast_window

    #get transactions of open order parts before observation date
    open_trans = trans_df.copy()[trans_df.index.isin(open_orders.index)] # part in open orders
    
    open_trans['trans_date'] = (open_trans['trans_date'] - obs_date).dt.days


    for row in open_orders.itertuples(): #group by orders to train as no trans history is valid input

        #move these outside loop with tails

        #future values (hidden from input layer only for deriving Y)
        all_this_trans = open_trans.loc[row.Index] # all transactions
        #get stock by end of the window
        transactions_during_window = all_this_trans[all_this_trans['trans_date'].between(0, forecast_days)]
        stck_added_during_window = transactions_during_window.loc[
            ((transactions_during_window['qty']>0) & # production (stock in)
             (transactions_during_window['wip']==0)& # for finished goods
             (transactions_during_window['correction']==0)), # not a stock correction
            'qty'].sum()

        #before observation date (accessible to input)
        this_trans = all_this_trans[(all_this_trans['trans_date'] < 0)] #and trans of this year if doing multiple years


        if this_trans.empty:
            transaction_predictors = [max_placeholder, 0,0,0,0,0,0,0]
        else:
            on_hand_qtys = list(this_trans[['on_hand', 'wip_on_hand']].iloc[-1])
            transaction_predictors = [
                this_trans['trans_date'].iloc[-1], #days since last transaction,
                len(this_trans),
                on_hand_qtys[0]/ row.qty_sum, # order_to_stck_ratio
                on_hand_qtys[0], #most recent stock on hand
                on_hand_qtys[1], #most recent wip on hand
                int(on_hand_qtys[0] >= row.qty_sum), #sufficient stock
                int(sum(on_hand_qtys) >= row.qty_sum), #sufficient_wip
                int(on_hand_qtys[0] - min(this_trans.loc[this_trans['wip'] == 0, 'on_hand'], default=0) >= row.qty_sum) #sufficient_without_stocktake
            ]

        dense_input = (
            list(row[2:]) #order data (not including target row[1])
            + transaction_predictors #  transaction data
            + date_meta # date data
            + stck_dict[row.Index] # part data
        )

        if row.desp_date_max == 0:
            y_label = y_labels.index('on_time')
        elif stck_added_during_window + on_hand_qtys[0] < row.qty_sum: #insufficient stock by end of window
            y_label = y_labels.index('no_stock')
        elif len(all_this_trans[ #if there exists a transaction for this product
                (all_this_trans['trans_date'].between(0, 10)) & # in the proceeding 10 days
                (all_this_trans['correction'] == 1) & # labelled as stock correction
                (all_this_trans['wip'] == 0) & # for finished goods
                (all_this_trans['qty'] < 0) # reducing stock
            ]) > 0:
            y_label = y_labels.index('stock_corrected')
        else:
            y_label = y_labels.index('missed')

        Y.append(y_label)
        X.append(this_trans.tail(512).to_numpy())
        dense.append(dense_input)
        stkno_ids.append(row.Index) # track stck_ids for later splitting

Y = np.array(Y)
stkno_ids = np.array(stkno_ids)

tprint(f'{len(Y)} observations gathered.')
print(np.bincount(Y.flatten()))


tprint('Padding X')
X, mask = pad_temporal_in(X)

tprint('Scaling data')
#scaling makes no difference to tree, does make difference to nn
X, x_scaler = scale(X, mask = mask)
dense, dense_scaler = scale(dense)

obs_dict = {}
tprint('Saving observations to cache')

for data, label in zip([mask, X, Y, dense, stkno_ids], ['mask', 'X', 'Y', 'dense', 'stkno_ids']):
    obs_dict[label] = data

np.savez_compressed(rf"cache\all_obs.npz", **obs_dict)

with open(r"cache\scalers.pkl", "wb") as f:
    pickle.dump({
        'dense_scaler': dense_scaler,
        'X_scaler': x_scaler
    }, f)
     
tprint('Done')

