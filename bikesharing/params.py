import os
import numpy as np

#### preprocessing params ####
START_YEAR = int(os.environ.get("START_YEAR"))
END_YEAR = int(os.environ.get("END_YEAR"))

############ GCP ##############
GCP_PROJECT = os.environ.get("GCP_PROJECT")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_REGION = os.environ.get("BQ_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")

############ MODEL ##############
FOLD_LENGTH = int(os.environ.get("FOLD_LENGTH"))
FOLD_STRIDE = int(os.environ.get("FOLD_STRIDE"))
TRAIN_TEST_RATIO = float(os.environ.get("TRAIN_TEST_RATIO"))
INPUT_LENGTH = int(os.environ.get("INPUT_LENGTH"))

########### CONSTANTS ###########
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "bikesharing", "data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "bikesharing", "training_outputs")
