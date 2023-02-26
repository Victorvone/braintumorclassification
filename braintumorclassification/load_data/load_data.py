import os
import numpy as np
import cv2
from google.cloud import storage


def load_data_gcloud(set):
    """
    loads data from the project's google cloud bucket
    """

    X, y = [], []
    target = 0
    folders = ["no_tumor", "meningioma_tumor", "glioma_tumor", "pituitary_tumor"]

    # Instantiates a client
    storage_client = storage.Client()

    # Get GCS bucket
    bucket = storage_client.get_bucket(os.environ.get("BUCKET"))

    for folder in folders:
        for image in list(bucket.list_blobs(prefix=f"{set}/{folder}")):
            picture = cv2.imdecode(
                np.frombuffer(image.download_as_bytes(), np.uint8), -1
            )
            picture_resized = cv2.resize(picture, (255, 255))
            X.append(picture_resized)
            y.append(target)
        target += 1

    return X, y


def load_data_local(directory, set):
    """
    loads data from the local directory
    """

    X, y = [], []
    directories = {}
    target = 0

    for foldername in os.listdir(f"{directory}/{set}"):
        if not foldername.startswith("."):
            directories[foldername] = f"{directory}/{set}/{foldername}"

    for tumor_directory in directories.values():
        for filename in os.listdir(tumor_directory):
            filepath = os.path.join(tumor_directory, filename)
            picture = cv2.imread(filepath)[:, :, :1]
            picture_resized = cv2.resize(picture, (255, 255))
            X.append(picture_resized)
            y.append(target)
        target += 1

    return X, y


def shuffle_and_format(X, y):
    ### shuffle data
    c = list(zip(X, y))
    np.random.shuffle(c)
    X, y = zip(*c)

    ### format data to array
    X = np.array(X)
    X = np.expand_dims(X, axis=3)
    y = np.array(y)

    return X, y


def make_data_sets(directory=None):
    source = os.environ.get("DATA_SOURCE")

    if source == "gcloud":
        X_train, y_train = load_data_gcloud("Training")
        X_test, y_test = load_data_gcloud("Testing")

    else:
        X_train, y_train = load_data_local(directory, "Training")
        X_test, y_test = load_data_local(directory, "Testing")

    X_train, y_train = shuffle_and_format(X_train, y_train)
    X_test, y_test = shuffle_and_format(X_test, y_test)

    print(
        f"X_train and y_train created successfully from {source} with shapes: \n X_train:{X_train.shape}\n y_train:{y_train.shape}"
    )
    print(
        f"X_test and y_test created successfully from {source} with shapes: \n X_test:{X_test.shape}\n y_test:{y_test.shape}"
    )

    return X_train, y_train, X_test, y_test
