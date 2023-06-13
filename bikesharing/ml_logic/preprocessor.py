from bikesharing.params import *
from bikesharing.ml_logic.encoders import *

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np



# Aggregate by hour
def group_rental_data_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups the rental data by hour.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with rental data grouped by hour.
    """
    # Preprocessing
    df['STARTTIME'] = pd.to_datetime(df['STARTTIME'])
    df['rent_date_hour'] = df['STARTTIME'].dt.floor('H')
    df['rent_date_hour'] = pd.to_datetime(df['STARTTIME']).dt.floor('H')


    # Grouping by Hour
    df_by_hour = df.groupby(by='rent_date_hour')[df.columns[1:-1]].sum()

    return df_by_hour.reset_index()


def preprocess_features(df: pd.DataFrame):
    scaler = MinMaxScaler()

    X = df[['temperature_2m', 'apparent_temperature','windspeed_10m', 'precipitation',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
            'weekday_sin' , 'weekday_cos']]

    X[scaler.get_feature_names_out] = scaler.fit_transform(X)

    return pd.concat([X , df[['is_weekend' , 'is_holiday']]])
