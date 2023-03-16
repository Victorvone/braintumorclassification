import streamlit as st
import numpy as np
import pydeck as pdk
import requests
from io import StringIO
from io import BytesIO
from PIL import Image
from PIL import ImageOps
import cv2
import tensorflow as tf
import os
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
import cv2
from scipy import stats
from tensorflow.keras.utils import to_categorical
#from tqdm import tqdm
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers, optimizers, callbacks
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import load_model
from PIL import Image
import requests
from io import BytesIO


st.title("Brain Tumor MRI Classification Example")

st.text("Upload a brain MRI Image for image classification as tumor or no-tumor")


from img_classification import teachable_machine_classification

uploaded_file = st.file_uploader("Choose a brain MRI ...", type="jpg")


if uploaded_file is not None:

        image = load_img(
            uploaded_file,
            grayscale=False,
            color_mode='rgb',
            target_size=(255, 255),
            interpolation='nearest',
            keep_aspect_ratio=False)

        st.image(image, caption='Uploaded MRI.', use_column_width=True)

        st.write("")

        st.write("Classifying...")

        model_EfficientNetv2 = '/home/aydogan/code/Victorvone/braintumorclassification/braintumorclassification/frontend/EfficientNetv2.h5'

        label = teachable_machine_classification(image, model_EfficientNetv2)

        if label == 0:

            st.write("Glioma has been detected")

        if label == 1:

            st.write("Meningioma has been detected")

        if label == 3:

            st.write("Pituitary has been detected")


        if label == 2:

            st.write("The MRI scan is healthy")
