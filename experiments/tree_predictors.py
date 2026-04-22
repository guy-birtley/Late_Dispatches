import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

#read db query output
orders_df = pd.read_pickle(r"cache\orders_df.pkl")

#set to datetime
for col in orders_df.columns:
    if col.split('_')[-1] in ('date', 'datetime'):
        orders_df[col] = pd.to_datetime(orders_df[col], errors="coerce")

#filter for years in 2024
orders_df = orders_df[orders_df['req_date'].dt.year.isin([2024, 2025])]

#extract leadtime parameter
orders_df['lead_time'] = (orders_df['req_date'] - orders_df['order_date']).dt.days

#sample 100 of each class
sampled_df = orders_df.groupby('late').sample(n=100, random_state=42)


clf = DecisionTreeClassifier(max_depth=2, criterion='entropy')
clf.fit(sampled_df[['qty', 'lead_time']], sampled_df['late'])

print(clf.tree_.feature, clf.tree_.threshold)
print(clf.tree_.value)


late_mask = (sampled_df['late'] == 1)
plt.scatter(sampled_df.loc[late_mask, 'qty'], sampled_df.loc[late_mask, 'lead_time'], label='Late', color = 'red')
plt.scatter(sampled_df.loc[~late_mask, 'qty'], sampled_df.loc[~late_mask, 'lead_time'], label='On Time', color = 'green')

#first tree split
plt.axvline(x=12.5, linestyle='--', color='black', label = 'First Split')

#second tree split
plt.hlines(y=7.5, xmin=0, xmax=12.5, linestyles='--', color = 'blue', label = 'Second Split')
plt.hlines(y=4.5, xmin=12.5, xmax=60, linestyles='--', color = 'blue')
plt.xlim(0, 60)
plt.ylim(0, 75)
plt.ylabel("Lead Time")
plt.xlabel("Order Qty")
plt.legend()
plt.show()
