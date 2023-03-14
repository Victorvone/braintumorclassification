import streamlit as st
import numpy as np
import pydeck as pdk
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
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers, optimizers, callbacks
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import load_model
import requests
import time

from img_classification import teachable_machine_classification

st.set_page_config(layout="wide", page_title="Brain Tumor Classification and Explainability App")

col1, col2, col3 = st.columns([6,6,6])

col1.markdown(" # Detect Brain MRI Scan")

uploaded_file = col1.file_uploader("Choose a brain MRI ...", type="jpg")
col2.success("MRI Scan was successfully uploaded!")
col2.image(uploaded_file, use_column_width=True)

progress_bar = col2.progress(0)

for perc_completed in range(100):
    time.sleep(0.005)
    progress_bar.progress(perc_completed+1)

with col3:
    col3.header("Tumor Class")

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        image = image.resize((255, 255))

        st.image(image, caption='Uploaded MRI.', use_column_width=True)

        st.write("")

        st.write("Classifying...")

        model_EfficientNetv2 = '/home/aydogan/code/Victorvone/braintumorclassification/braintumorclassification/frontend/EfficientNetv2.h5'

        label = teachable_machine_classification(image, model_EfficientNetv2)

        if label == 0:

            st.success("Meningioma has been detected!", icon='✅')

        if label == 1:

            st.write("Meningioma has been detected")

        if label == 3:

            st.write("Pituitary has been detected")


        if label == 2:

            st.write("The MRI scan is healthy")
