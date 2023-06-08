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
    df['rent_date_hour'] = df['STARTTIME'].dt.floor('H')
    df['rent_date_hour'] = pd.to_datetime(df['STARTTIME']).dt.floor('H')


    # Grouping by Hour
    df_by_hour = df.groupby(by='rent_date_hour')[df.columns[1:-1]].sum()

    return df_by_hour


def preprocess_features(X: pd.DataFrame) -> np.ndarray:
    def create_preprocessor() -> ColumnTransformer:

        # SCALE PIPE
        scaler_pipe = Pipeline([
            ('scaler', MinMaxScaler())
        ])

        return scaler_pipe


    preprocessor = create_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    print("✅ X_processed, with shape", X_processed.shape)

    return X_processed
