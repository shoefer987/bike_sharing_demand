import pandas as pd
import numpy as np
import csv, json
import geopandas as gpd
from shapely.geometry import Polygon, Point
import requests
import holidays

from bikesharing.params import *
from bikesharing.ml_logic.data import get_raw_data

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler


from bikesharing.ml_logic.encoders import *



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
