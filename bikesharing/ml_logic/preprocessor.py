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

    df = df.fillna(0)
    def create_preprocessor() -> ColumnTransformer:

        # SCALE PIPE
        scaler_pipe = Pipeline([
            ('scaler', MinMaxScaler())
        ])

        return scaler_pipe

    X = df[['temperature_2m', 'relativehumidity_2m', 'apparent_temperature',
       'windspeed_10m', 'precipitation','hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos']]

    preprocessor = create_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    return pd.concat([pd.DataFrame(X_processed) , df[['is_holiday', 'is_weekend']]] , axis=1) , y
