from datetime import datetime

y_labels = ['on_time', 'no_stock', 'stock_corrected', 'missed']


def tprint(text):
    print(datetime.now(), text)