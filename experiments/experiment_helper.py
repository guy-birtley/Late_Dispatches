import pandas as pd

#read db query output
orders_df = pd.read_pickle(r"cache\orders_df.pkl")

#set to datetime
for col in orders_df.columns:
    if col.split('_')[-1] in ('date', 'datetime'):
        orders_df[col] = pd.to_datetime(orders_df[col], errors="coerce")