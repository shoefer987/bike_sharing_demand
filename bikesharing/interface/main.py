import pandas as pd

from bikesharing.ml_logic.data import get_raw_data, get_weather_data
from bikesharing.params import *

from pathlib import Path
from colorama import Fore, Style



def preprocess() -> pd.DataFrame:
    """
    1. call get_raw_data
    2. drop cols
    3. clean (rm duplicates)
    3. encode y
    4. aggregate by hour
    5. join with weather data (get_weather_data)
    6. feature engineering & merge
    7. feature selection
    8. preproc-pipeline
    """

    cache_path_preproc=Path(f'{LOCAL_DATA_PATH}/processed/processed_from_{START_YEAR}_to_{END_YEAR}.csv')

    if cache_path_preproc.is_file():
        print(Fore.BLUE + "\nLoad preprocessed data from local CSV..." + Style.RESET_ALL)
        preproc_df = pd.read_csv(cache_path_preproc, header='infer')
        return preproc_df

    print(Fore.BLUE + "\nPreprocessing Data..." + Style.RESET_ALL)

    # 1. get_raw_data
    query =f'''
        SELECT *
        FROM `{GCP_PROJECT}.{BQ_DATASET}.raw_data_mvg`
    '''

    rental_data_df = get_raw_data(gcp_project=GCP_PROJECT , query=query , cache_path=Path(f'{LOCAL_DATA_PATH}/raw/mvg_rentals_from_{START_YEAR}_to_{END_YEAR}.csv'))

    # 2. drop cols
    rental_relavent_cols_df = rental_data_df[['STARTTIME' , 'STARTLAT' , 'STARTLON']]

    # 3. clean(rm duplicates)
    rental_relavent_cols_df = rental_relavent_cols_df.drop_duplicates()

    # 4. encode y
        # encoded_rental_df = encode_y(rental_data_relcols_df)

    # 5. aggregate by hour
        # aggregated_rental_df = aggregate_by_hour(encoded_rental_df)

    # 6. join with weather data
    weather_data_df = get_weather_data(cache_path=Path(f'{LOCAL_DATA_PATH}/raw/histotical_weather_data_{START_YEAR}_to_{END_YEAR}.csv'))
        # merged_df = aggregated_rental_df.merge(weather_data_df, right_on='<time_col_weather>' , left_on='<time_col_rental>' , how=left)

    # 7. feature enginering & merge
        # holidays = flag_holidays(<time_col>)
        # merged_df = merged_df.merge(holidays , on='<time_col>' , how='inner')
        # weekends = flag_weekends(<time_col>)
        # merged_df = merged_df.merge(weekends , on='<time_col>' , how='inner')
        # encoded_date = encode_time(<time_col>)
        # merged_df = merged_df.merge(encoded_date , on='<time_col>' , how='inner')

    # 8. feature selection
        # selected_merged_df = select_features(merged_df , <List_of_features>)

    # 9. preproc-pipeline
        # preproc_df = pd.Dataframe(preproc_pipeline(selected_merged_df))

    return preproc_df

# function to be defined
def train():
    """
    - Download processed data from your BQ table (or from cache if it exists)
    - Train on the preprocessed dataset (which should be ordered by date)
    - Store training results and model weights

    Return val_mae as a float
    """

    # 1. Load processed data
        # 1.1 Load from cache if present

        # 1.2 Load from GBQ if not

    # 2. Test/Train/Val Split
        # Train=2 years
        # Test=1 year
    pass

# function to be defined
def predict():
    pass

# function to be defined
def evaluate():
    pass
