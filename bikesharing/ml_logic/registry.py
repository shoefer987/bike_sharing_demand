import os, time, glob
from bikesharing.params import *

import joblib

from colorama import Fore, Style


def save_model(model , district : str) -> None:
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{district}_{timestamp}.pkl")
    joblib.dump(model , model_path)

    print("✅ Model saved locally")
    return

def load_model():
    print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

    # Get the latest model version name by the timestamp on disk
    local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
    local_model_paths = glob.glob(f"{local_model_directory}/*")

    if not local_model_paths:
        return None

    most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

    print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

    latest_model = joblib.load(most_recent_model_path_on_disk)

    print("✅ Model loaded from local disk")

    return latest_model
