import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

from sklearn.preprocessing import OneHotEncoder


def get_district_from_polygons(rental_df: pd.DataFrame, polygons: dict) -> pd.DataFrame:
    """
    Performs a spatial join between the rental DataFrame and polygons.

    Args:
        rental_df (pd.DataFrame): The rental DataFrame.
        polygons (dict): The dictionary of polygons.

    Returns:
        pd.DataFrame: The DataFrame with the spatial join result.
    """
    # Create a DataFrame from the polygons dictionary
    polygons_df = pd.DataFrame.from_dict(polygons, orient='index', columns=['geometry'])
    # Reset the index to make the 'district' column a regular column
    polygons_df = polygons_df.reset_index().rename(columns={'index': 'district'})

    # Create a GeoDataFrame from the polygons DataFrame
    polygons_gdf = gpd.GeoDataFrame(polygons_df)
    # Set the geometry column in the polygons_gdf GeoDataFrame
    polygons_gdf.set_geometry('geometry', inplace=True)

    # Create a GeoDataFrame from the point data
    geometry = [Point(row['STARTLON'], row['STARTLAT']) for _, row in rental_df.iterrows()]
    rental_gdf = gpd.GeoDataFrame(rental_df, geometry=geometry)
    # Set the geometry column in the rental_gdf GeoDataFrame
    rental_gdf.set_geometry('geometry', inplace=True)

    # Perform the spatial join
    rental_geo_df = gpd.sjoin(rental_gdf, polygons_gdf, predicate='within')

    # Drop unnecessary columns
    rental_geo_df = rental_geo_df.drop(columns=['geometry', 'index_right', 'STARTLON', 'STARTLAT'])

    return rental_geo_df



def encode_district_label(rental_df: pd.DataFrame, polygons: dict) -> pd.DataFrame:
    """
    Encodes the district labels in the DataFrame using one-hot encoding.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with encoded district labels.
    """
    df = get_district_from_polygons(rental_df, polygons)

    # Instantiate the OneHotEncoder
    district_ohe = OneHotEncoder(sparse_output=False)

    # Fit encoder
    district_ohe.fit(df[['district']])

    # Apply one-hot encoding and add the encoded columns to the DataFrame
    encoded_columns = district_ohe.get_feature_names_out()
    encoded_values = district_ohe.transform(df[['district']])
    df[encoded_columns] = encoded_values
    df.drop(columns=['district'] , inplace=True)

    # Update the column names in df without the prefix 'district_'
    column_names = [column.split('district_', 1)[-1] for column in df.columns]
    df.columns = column_names

    return df


def encode_temporal_features(datetime_column: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes temporal features (hour, month, day) in the DataFrame using sine and cosine transformations.

    Args:
        datetime_column (pd.Series): The input series with datetime values.

    Returns:
        pd.DataFrame: The DataFrame with encoded temporal features.
    """
    # Create a new DataFrame to hold the encoded features
    encoded_df = pd.DataFrame()
    encoded_df['rent_date_hour'] = datetime_column['rent_date_hour']

    # Extract hour, month, and day from the datetime_column
    encoded_df['hour'] = datetime_column['rent_date_hour'].dt.hour.apply(lambda x: x+1)
    encoded_df['month'] = datetime_column['rent_date_hour'].dt.month
    encoded_df['day'] = datetime_column['rent_date_hour'].dt.day

    temporal_features = ['hour', 'month', 'day']

    # Apply sine and cosine transformations to the temporal features
    for feature in temporal_features:
        encoded_df[f'{feature}_sin'] = np.sin(2 * np.pi * encoded_df[feature] / encoded_df[feature].max())
        encoded_df[f'{feature}_cos'] = np.cos(2 * np.pi * encoded_df[feature] / encoded_df[feature].max())

    encoded_df.drop(columns=temporal_features, inplace=True)

    return encoded_df
