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
from requests_toolbelt.multipart.encoder import MultipartEncoder
import io


from img_classification import teachable_machine_classification

# interact with FastAPI endpoint
from requests_toolbelt.multipart.encoder import MultipartEncoder

backend = "http://127.0.0.1:8000/predict4"
backend2 = "http://127.0.0.1:8000/predict5"

# Function for fastapi interface

def process(image, server_url: str):

    m = MultipartEncoder(fields={"file": ("filename", image, "image/jpeg")})

    r = requests.post(
        server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000
    )
    return r







#im = Image.open('/home/ivana/code/Victorvone/braintumorclassification/braintumorclassification/frontend/brain.png')
#page_icon = im
st.set_page_config(layout="wide", page_title="Brain Tumor Classification and Explainability App")

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

            # Result of the interface
            segments = process(uploaded_file,backend)
            segments2 = process(uploaded_file,backend2)
            exp_image = Image.open(io.BytesIO(segments.content)).convert("RGB")
            result_string = segments2.text

            st.image(exp_image, use_column_width=True)

            st.write("")
            st.write("")


            st.error(result_string, icon ='☠')

            #st.write("Classifying...")

            # model_EfficientNetv2 = '/home/aydogan/code/Victorvone/braintumorclassification/braintumorclassification/frontend/EfficientNetv2.h5'

            # label = teachable_machine_classification(image, model_EfficientNetv2)

            # if label == 0:

            #     st.error("Meningioma has been detected!", icon ='☠')

            # if label == 1:

            #     st.error("Meningioma has been detected", icon ='☠')

            # if label == 3:

            #     st.error("Pituitary has been detected", icon ='☠')


            # if label == 2:

            #     st.success("The MRI scan is healthy")

with tab2:

    st.title('Meet the Team')

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        #st.header("Aydogan")
        #st.write(os.getcwd())


        dirname = os.path.dirname(__file__) #this gets current directory you placed your applications
        image = Image.open(dirname +'/aydogan.JPG')

        # Create container with centered image
        with st.container():
            st.markdown("<h1 style='text-align: center'>Aydogan</h1>", unsafe_allow_html=True)
            col1.info('Researcher on Geospatial Analysis')
            st.markdown('[Github](https://github.com/aydogan22)')
            st.image(image, use_column_width=True)


    with col2:
        image2 = Image.open(dirname +'/victor.JPG')

        #st.header("Victor")

        #st.image(image2, width=250)

        with st.container():
            st.markdown("<h1 style='text-align: center'>Victor</h1>", unsafe_allow_html=True)
            col2.info('Add Info')
            st.markdown('[Github](https://github.com/Victorvone)')
            st.image(image2, use_column_width=True)

    with col3:
        #st.header("Aurélien Biais")


        image3 = Image.open(dirname +'/aurelien.jpg')



        with st.container():
            st.markdown("<h1 style='text-align: center'>Aurélien</h1>", unsafe_allow_html=True)
            col3.write("")
            col3.write("")
            col3.write("")
            col3.info('Data Analytics Lead @recare')
            st.markdown('[Github](https://github.com/abiais)')
            st.image(image3, use_column_width=True)
        #st.image(image3, width=250)
    with col4:
        #st.header("Aurélien Biais")
        image4 = Image.open(dirname +'/ivan.JPG')

        with st.container():
            st.markdown("<h1 style='text-align: center'>Ivan</h1>", unsafe_allow_html=True)
            col4.info('Add Info')
            st.markdown('[Github](https://github.com/IvanAndjelkovic)')
            st.image(image4, use_column_width=True)
        #st.image(image4, width=250)
