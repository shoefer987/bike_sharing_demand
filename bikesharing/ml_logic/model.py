import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Sequence

def get_folds(
    df: pd.DataFrame,
    fold_length: int,
    fold_stride: int) -> List[pd.DataFrame]:
    """
    This function slides through the Time Series dataframe of shape (n_timesteps, n_features) to create folds
    - of equal `fold_length`
    - using `fold_stride` between each fold

    Args:
        df (pd.DataFrame): Overall dataframe
        fold_length (int): How long each fold should be in rows
        fold_stride (int): How many timesteps to move forward between taking each fold

    Returns:
        List[pd.DataFrame]: A list where each fold is a dataframe within
    """
    folds = []
    for idx in range(0, len(df), fold_stride):
        # Exits the loop as soon as the last fold index would exceed the last index
        if (idx + fold_length) > len(df):
            break
        fold = df.iloc[idx:idx + fold_length, :]
        folds.append(fold)
    return folds

def train_test_indices(fold:pd.DataFrame,
                    train_test_ratio: float,
                    input_length: int) -> Tuple[pd.DataFrame]:
    """From a fold dataframe, take a train dataframe and test dataframe based on
    the split ratio.
    - df_train contains all the timesteps until round(train_test_ratio * len(fold))
    - df_test contains all the timesteps needed to create all (X_test, y_test) tuples

    Args:
        fold (pd.DataFrame): A fold of timesteps
        train_test_ratio (float): The ratio between train and test 0-1
        input_length (int): How long each X_i will be

    Returns:
        Tuple[pd.DataFrame]: A tuple of two dataframes (fold_train, fold_test)
    """

def train_test_split(fold:pd.DataFrame,
                     train_test_ratio: float,
                     input_length: int) -> Tuple[pd.DataFrame]:
    """From a fold dataframe, take a train dataframe and test dataframe based on
    the split ratio.
    - df_train should contain all the timesteps until round(train_test_ratio * len(fold))
    - df_test should contain all the timesteps needed to create all (X_test, y_test) tuples

    Args:
        fold (pd.DataFrame): A fold of timesteps
        train_test_ratio (float): The ratio between train and test 0-1
        input_length (int): How long each X_i will be

    Returns:
        Tuple[pd.DataFrame]: A tuple of two dataframes (fold_train, fold_test)
    """

    # TRAIN SET
    last_train_idx = round(train_test_ratio * len(fold))
    fold_train = fold.iloc[0:last_train_idx, :]

    # TEST SET
    first_test_idx = last_train_idx - input_length
    fold_test = fold.iloc[first_test_idx:, :]

    return (fold_train, fold_test)

def get_Xi_yi(
    fold:pd.DataFrame,
    input_length:int,
    output_length:int) -> Tuple[pd.DataFrame]:
    """given a fold, it returns one sequence (X_i, y_i) as based on the desired
    input_length and output_length with the starting point of the sequence being chosen at random based

    Args:
        fold (pd.DataFrame): A single fold
        input_length (int): How long each X_i should be
        output_length (int): How long each y_i should be

    Returns:
        Tuple[pd.DataFrame]: A tuple of two dataframes (X_i, y_i)
    """

    first_possible_start = 0
    last_possible_start = len(fold) - (input_length + output_length) + 1
    random_start = np.random.randint(first_possible_start, last_possible_start)
    X_i = fold.iloc[random_start:random_start+input_length]
    y_i = fold.iloc[random_start+input_length:
                  random_start+input_length+output_length][[TARGET]]

    return (X_i, y_i)

def get_X_y(
    fold:pd.DataFrame,
    number_of_sequences:int,
    input_length:int,
    output_length:int) -> Tuple[np.array]:
    """Given a fold generate X and y based on the number of desired sequences
    of the given input_length and output_length

    Args:
        fold (pd.DataFrame): Fold dataframe
        number_of_sequences (int): The number of X_i and y_i pairs to include
        input_length (int): Length of each X_i
        output_length (int): Length of each y_i

    Returns:
        Tuple[np.array]: A tuple of numpy arrays (X, y)
    """
    X, y = [], []

    for i in range(number_of_sequences):
        (Xi, yi) = get_Xi_yi(fold, input_length, output_length)
        X.append(Xi)
        y.append(yi)

    return np.array(X), np.array(y)

def get_model_params(district:str) -> dict:
    '''
        returns the hyperparameters for training an XGBRegressor for the
        specified district
    '''
    hyperparams = {
        'Feldmoching': {
            'n_estimators': 100,
            'max_depth': 5
        },
        'Obersendling': {
            'n_estimators': 100,
            'max_depth': 5,
            'colsample_bytree': 0.8,
            'eta': 0.1,
            'gamma': 5,
            'min_child_weight': 3
        },
        'Ludwigsvorstadt-Isarvorstadt': {
            'n_estimators': 10,
            'max_depth': 7,
            'eta': 0.3,
            'min_child_weight': 5,
            'tree_method': 'approx'
        },
        'Neuhausen-Nymphenburg': {
            'n_estimators': 10,
            'max_depth': 6,
            'min_child_weight': 3,
            'tree_method': 'approx'
        },
        'Sendling': {
            'n_estimators': 100,
            'max_depth': 5
        },
        'Maxvorstadt': {
            'n_estimators': 10,
            'max_depth': 10,
            'min_child_weight': 2,
            'tree_method': 'approx'
        },
        'Untergiesing-Harlaching': {
            'n_estimators': 10,
            'max_depth': 6,
            'eta': 0.3,
            'min_child_weight': 5,
            'tree_method': 'approx'
        },
        'Laim': {
            'max_depth': 7,
            'colsample_bytree': 0.6,
            'eta': 0.05,
            'gamma': 0.5,
            'min_child_weight': 3,
            'subsample': 1
        }
    }

    if district in ['Obersendling', 'Hadern', 'Pasing-Obermenzing', 'Aubing-Lochhausen-Langwied',
                    'Thalkirchen', 'Pasing', 'Trudering-Riem', 'Harlaching', 'Hasenbergl-Lerchenau Ost',
                    'Südgiesing', 'Obermenzing', 'Trudering']:
        return hyperparams['Obersendling']
    elif district in ['Laim', 'Obergiesing', 'Sendling-Westpark', 'Ramersdorf-Perlach', 'Berg am Laim']:
        return hyperparams['Laim']
    elif district in ['Untergiesing-Harlaching', 'Bogenhausen', 'Untergiesing', 'Schwanthalerhöhe', 'Berg am Laim']:
        return hyperparams['Untergiesing-Harlaching']
    elif district in ['Sendling', 'Schwabing-West', 'Moosach', 'Au - Haidhausen']:
        return hyperparams['Sendling']
    elif district in ['Feldmoching', 'Untermenzing-Allach', 'Lochhausen']:
        return hyperparams['Feldmoching']
    elif district in ['Neuhausen-Nymphenburg', 'Milbertshofen-Am Hart']:
        return hyperparams['Neuhausen-Nymphenburg']
    elif district in ['Ludwigsvorstadt-Isarvorstadt', 'Schwabing-Freimann']:
        return hyperparams['Ludwigsvorstadt-Isarvorstadt']
    elif district in ['Maxvorstadt', 'Altstadt-Lehel']:
        return hyperparams['Maxvorstadt']
    else:
        return None
