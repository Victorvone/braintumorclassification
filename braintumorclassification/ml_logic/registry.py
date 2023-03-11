import mlflow
from mlflow.keras import log_model

import glob2
import time
import os
import pickle

from colorama import Fore, Style

from tensorflow.keras import Model, models

import gdown


def save_model(model: Model = None, params: dict = None, metrics: dict = None) -> None:
    """
    persist trained model, params and metrics
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    if os.environ.get("MODEL_TARGET") == "mlflow":
        print(Fore.BLUE + "\nSave model to Mlflow..." + Style.RESET_ALL)

        # retrieve mlflow env params
        mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        mlflow_experiment = os.environ.get("MLFLOW_EXPERIMENT")

        # configure mlflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name=mlflow_experiment)

        with mlflow.start_run():
            # STEP 1: push parameters to mlflow
            if params is not None:
                mlflow.log_params(params)

            # STEP 2: push metrics to mlflow
            if metrics is not None:
                mlflow.log_metrics(metrics)

            # STEP 3: push model to mlflow
            if model is not None:
                log_model(
                    model=model,
                    artifact_path="dummy_model",
                    registered_model_name="braintumorclassification",
                )

        print("\n✅ model saved to Mlflow")
        return None

    print(Fore.BLUE + "\nSave model to local disk..." + Style.RESET_ALL)

    # save params
    if params is not None:
        params_path = os.path.join(
            os.environ.get("LOCAL_REGISTRY_PATH"), "params", timestamp + ".pickle"
        )
        print(f"- params path: {params_path}")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # save metrics
    if metrics is not None:
        metrics_path = os.path.join(
            os.environ.get("LOCAL_REGISTRY_PATH"), "metrics", timestamp + ".pickle"
        )
        print(f"- metrics path: {metrics_path}")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    # save model
    if model is not None:
        model_path = os.path.join(
            os.environ.get("LOCAL_REGISTRY_PATH"), "models", timestamp
        )
        print(f"- model path: {model_path}")
        model.save(model_path)

    print("\n✅ model saved to disk")
    return None


def load_model() -> Model:
    """
    load the latest saved model, return None if no model found
    """

    # load model from mlflow
    if os.environ.get("MODEL_TARGET") == "mlflow":
        stage = "Production"

        print(
            Fore.BLUE + f"\nLoad model {stage} stage from mlflow..." + Style.RESET_ALL
        )

        mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

        mlflow.set_tracking_uri(mlflow_tracking_uri)

        model_uri = f"models:/braintumorclassification/{stage}"

        model = mlflow.keras.load_model(model_uri=model_uri)

        print("\n✅ model loaded from Mlflow")
        return model

    # load model from gdrive
    if os.environ.get("MODEL_TARGET") == "gdrive":
        stage = "Production"

        print(Fore.BLUE + f"\nLoad model {stage} stage from gdriv..." + Style.RESET_ALL)

        url = "https://drive.google.com/file/d/1-2UUABb2fBQwCL8rpigP6cnOJlGZcGDL/view?usp=share_link"
        output_path = "./models/ResNet50v2.h5"
        gdown.download(url, output_path, quiet=False, fuzzy=True)
        model = models.load_model(output_path, compile=False)

        print("\n✅ model loaded from gdrive")

        return model

    print(Fore.BLUE + "\nLoad model from local disk..." + Style.RESET_ALL)

    # get latest model version
    model_directory = os.path.join(os.environ.get("LOCAL_REGISTRY_PATH"), "models")

    results = glob2.glob(f"{model_directory}/*")
    if not results:
        return None

    model_path = sorted(results)[-1]
    print(f"- path: {model_path}")

    model = models.load_model(model_path)
    print("\n✅ model loaded from disk")

    return model
