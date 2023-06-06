import numpy as np
import pandas as pd
from colorama import Fore, Style
from pathlib import Path

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
