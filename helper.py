
import pandas as pd
from sqlalchemy import create_engine


forecast_horizon = 2



def tprint(text):
    print(pd.Timestamp.now(), text)