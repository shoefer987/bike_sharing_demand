import os
import numpy as np

#### preprocessing params ####
START_YEAR = int(os.environ.get("START_YEAR"))
END_YEAR = int(os.environ.get("END_YEAR"))

########### GCP ##############
GCP_PROJECT = os.environ.get("GCP_PROJECT")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_REGION = os.environ.get("BQ_REGION")

########### CONSTANTS ###########
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "bikesharing", "data")
