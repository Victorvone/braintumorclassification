import keras

from PIL import Image, ImageOps

import numpy as np

import tensorflow as tf

from tensorflow.keras.models import load_model

from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img




def teachable_machine_classification(img, weights_file):

    """It returns the position of the highest probability then we gonna use it in the app.py"""

    # Load the model

    model = keras.models.load_model(weights_file)

    test_img_ndarray = img_to_array(
    img)

    test_img_ndarray_exp = np.expand_dims(test_img_ndarray, axis=0)
    res = model.predict(test_img_ndarray_exp)

    return np.argmax(res) # return position of the highest probability
