import numpy as np
from tensorflow.keras.utils import to_categorical


def preprocess(X_train, y_train, X_test, y_test):

    'preprocess data'
    # Normalizing
    X_train_preproc = X_train / 255
    X_test_preproc = X_test / 255

    # OneHotEncoding
    y_train_preproc = to_categorical(y_train, 4)
    y_test_preproc = to_categorical(y_test, 4)

    return X_train_preproc, y_train_preproc, X_test_preproc, y_test_preproc
