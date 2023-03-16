import streamlit as st
import numpy as np
import pydeck as pdk
from io import StringIO
from io import BytesIO
from PIL import Image, ImageDraw
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

im = Image.open('/home/aydogan/code/Victorvone/braintumorclassification/braintumorclassification/frontend/brain.png')
st.set_page_config(layout="wide", page_title="Brain Tumor Classification and Explainability App", page_icon = im)

tab1, tab2= st.tabs(["Application", "Team"])

with tab1:




    col1, col2, col3 = st.columns([6,6,6])
    col1.write("")
    col1.write("")
    col1.write("")
    col1.write("")
    col1.write("")
    col1.write("")
    col1.write("")
    col1.write("")
    col1.write("")
    col1.write("")
    col1.write("")
    col1.write("")
    col1.write("")

    col1.title("Choose a brain MRI ...")
    uploaded_file = col1.file_uploader("", type="jpg")
    if uploaded_file:

        # Hide filename on UI
        st.markdown('''
            <style>
                .uploadedFile {display: none}
            <style>''',
            unsafe_allow_html=True)
        col2.header("Brain Scan")
        col2.write("")
        col2.write("")

        col2.image(uploaded_file, use_column_width=True)
        progress_bar = col2.progress(0)

        for perc_completed in range(100):
            time.sleep(0.005)
            progress_bar.progress(perc_completed+1)

        col2.write("")
        col2.info("MRI Scan was successfully uploaded!", icon='✔')


        col3.header("Tumor Class")


    with col3, col3.expander('Click Here to Classify MRI Scan!'):


        if uploaded_file is not None:

            image = Image.open(uploaded_file)
            image = image.resize((255, 255))

            st.image(image, use_column_width=True)

            st.write("")
            st.write("")


            #st.write("Classifying...")

            model_EfficientNetv2 = '/home/aydogan/code/Victorvone/braintumorclassification/braintumorclassification/frontend/EfficientNetv2.h5'

            label = teachable_machine_classification(image, model_EfficientNetv2)

            if label == 0:

                st.error("Meningioma has been detected!", icon ='☠')

            if label == 1:

                st.error("Meningioma has been detected", icon ='☠')

            if label == 3:

                st.error("Pituitary has been detected", icon ='☠')


            if label == 2:

                st.success("The MRI scan is healthy")

with tab2:

    st.title('Meet the Team')

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        #st.header("Aydogan")

        image = Image.open('/home/aydogan/code/Victorvone/braintumorclassification/braintumorclassification/frontend/aydogan.JPG')

        # Create container with centered image
        with st.container():
            st.markdown("<h1 style='text-align: center'>Aydogan</h1>", unsafe_allow_html=True)
            col1.info('Geospatial data scientist')
            st.markdown('[Github](https://github.com/aydogan22)')
            st.image(image, use_column_width=True)


    with col2:
        image2 = Image.open('/home/aydogan/code/Victorvone/braintumorclassification/braintumorclassification/frontend/victor.JPG')
        #st.header("Victor")

        #st.image(image2, width=250)

        with st.container():
            st.markdown("<h1 style='text-align: center'>Victor</h1>", unsafe_allow_html=True)
            st.image(image2, use_column_width=True)
        col2.info('Add Info')
    with col3:
        #st.header("Aurélien Biais")
        col3.write("")
        col3.write("")
        col3.write("")
        image3 = Image.open('/home/aydogan/code/Victorvone/braintumorclassification/braintumorclassification/frontend/aurelien.jpg')
        with st.container():
            st.markdown("<h1 style='text-align: center'>Aurélien Biais</h1>", unsafe_allow_html=True)
            st.image(image3, use_column_width=True)
        col3.info('Data Analytics Lead @recare')
        st.markdown('[Recare](https://github.com/recare)')
        #st.image(image3, width=250)
    with col4:
        #st.header("Aurélien Biais")
        image4 = Image.open('/home/aydogan/code/Victorvone/braintumorclassification/braintumorclassification/frontend/ivan.JPG')
        with st.container():
            st.markdown("<h1 style='text-align: center'>Ivan Andjelkovic</h1>", unsafe_allow_html=True)
            st.image(image4, use_column_width=True)
        #st.image(image4, width=250)
        col4.info('Add Info')
