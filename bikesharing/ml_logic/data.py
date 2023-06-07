import numpy as np
import pandas as pd
from colorama import Fore, Style
from pathlib import Path
import requests

from bikesharing.params import *

from google.cloud import bigquery


def get_raw_data():
    dfs = []
    for year in range(START_YEAR,END_YEAR+1,1):
        df = pd.read_csv(f'raw_data/MVG_Rad_Fahrten_{year}.csv', sep=';')
        cols = [col.strip() for col in df.columns]
        df.columns = cols
        dfs.append(df)

    return pd.concat(dfs, axis=0)

def get_raw_data(
        gcp_project:str,
        query:str,
        cache_path:Path,
        data_has_header=True
    ) -> pd.DataFrame:
    """
    Retrieve `query` data from BigQuery, or from `cache_dir` if a file exists
    Store in `cache_dir` if retrieved from BigQuery for future use
     * cache_path: the path where to look for (or store) the cached data,
            e. g.: cache_path = Path(LOCAL_DATA_PATH).joinpath("raw",
                                f"raw_{START_YEAR}_{END_YEAR}.csv")
     * query: the string containing the query which should be run on the table
            e. g.: query = f'''
                        SELECT *
                        FROM `{GCP_PROJECT}.{BQ_DATASET}.raw_data_mvg`
                        '''
    """

    if cache_path.is_file():
        print(Fore.BLUE + "\nLoad data from local CSV..." + Style.RESET_ALL)
        df = pd.read_csv(cache_path, header='infer' if data_has_header else None)
    else:
        print(Fore.BLUE + "\nLoad data from BigQuery server..." + Style.RESET_ALL)
        client = bigquery.Client(project=gcp_project)
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()

        # Store as CSV if the BQ query returned at least one valid line
        if df.shape[0] > 1:
            df.to_csv(cache_path, header=data_has_header, index=False)
            print(f'columns: {df.columns}')

    print(f"✅ Data loaded, with shape {df.shape}")

    return df

def get_weather_data(
        cache_path:Path,
        data_has_header=True):
    """
    Retrieve the historical weather data from 'start_year' to 'end_year' from the
    Open Meteo Api.
    """

    if cache_path.is_file():
        print(Fore.BLUE + "\nLoad data from local CSV..." + Style.RESET_ALL)
        historical_weather_data_df = pd.read_csv(cache_path, header='infer' if data_has_header else None)
    else:
        # URL: https://archive-api.open-meteo.com/v1/archive?latitude=48.70&longitude=13.46&start_date=2019-01-01&end_date=2022-12-31&hourly=temperature_2m,relativehumidity_2m,apparent_temperature,precipitation,windspeed_10m

        base_url = 'https://archive-api.open-meteo.com/v1/archive'

        params = {
            'latitude': 48.70,
            'longitude': 13.46,
            'start_date' : f'{START_YEAR}-01-01',
            'end_date' : f'{END_YEAR}-12-31',
            'hourly': ['temperature_2m', 'relativehumidity_2m', 'apparent_temperature','windspeed_10m','precipitation']
        }

        historical_weather_data = requests.get(base_url , params=params).json()

        # check keys for hourly

        historical_weather_data_df = pd.DataFrame(historical_weather_data['hourly'])

        if historical_weather_data_df.shape[0] > 1:
            historical_weather_data_df.to_csv(cache_path, header=data_has_header, index=False)
            print(f'columns: {historical_weather_data_df.columns}')

    print(f"✅ Data loaded, with shape {historical_weather_data_df.shape}")

    return historical_weather_data_df
