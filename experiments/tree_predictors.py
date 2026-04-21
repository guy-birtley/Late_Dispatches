import pandas as pd
import matplotlib.pyplot as plt

orders_df = pd.read_pickle(r"cache\orders_df.pkl")

for col in orders_df.columns:
    if col.split('_')[-1] in ('date', 'datetime'):
        orders_df[col] = pd.to_datetime(orders_df[col], errors="coerce")

orders_df = orders_df[orders_df['req_date'].dt.year.isin([2024, 2025])]


orders_df['lead_time'] = (orders_df['req_date'] - orders_df['order_date']).dt.days

sampled_df = orders_df.groupby('late').sample(n=100, random_state=42)

for i, label in enumerate(['On Time', 'Late']):
    mask = (sampled_df['late'] == i)
    plt.scatter(sampled_df.loc[mask, 'qty'],sampled_df.loc[mask, 'lead_time'], label=label)

plt.ylabel("Lead Time")
plt.xlabel("Order Qty")
plt.legend()
plt.show()


plt.show()