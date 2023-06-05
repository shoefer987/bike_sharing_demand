import numpy as np
import pandas as pd
from bikesharing.params import *

def get_raw_data():
    dfs = []
    for year in range(START_YEAR,END_YEAR+1,1):
        df = pd.read_csv(f'raw_data/MVG_Rad_Fahrten_{year}.csv', sep=';')
        cols = [col.strip() for col in df.columns]
        df.columns = cols
        dfs.append(df)

    return pd.concat(dfs, axis=0)
