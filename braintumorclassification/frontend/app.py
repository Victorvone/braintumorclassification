import streamlit as st

from PIL import Image
import os
# from tqdm import tqdm
import requests
import time
from requests_toolbelt.multipart.encoder import MultipartEncoder
import io
# interact with FastAPI endpoint


backend = "https://btc-w3zkxihrsq-ey.a.run.app/predict4"
backend2 = "https://btc-w3zkxihrsq-ey.a.run.app/predict5"

# Function for fastapi interface


def process(image, server_url: str):
    m = MultipartEncoder(fields={"file": ("filename", image, "image/jpeg")})

    r = requests.post(
        server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000
    )
    return r


dirname = os.path.dirname(
    __file__
)  # this gets current directory you placed your applications
im = Image.open(dirname + "/brain.png")

st.set_page_config(
    layout="wide",
    page_title="Brain Tumor Classification and Explainability App",
    page_icon=im,
)

tab1, tab2 = st.tabs(["💻 Application", "👉Team"])

with tab1:
    col1, col2, col3 = st.columns([6, 6, 6])
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

    col1.title("Upload a brain MRI 📥")
    uploaded_file = col1.file_uploader("", type="jpg")
    if uploaded_file:
        # Hide filename on UI
        st.markdown(
            """
            <style>
                .uploadedFile {display: none}
            <style>""",
            unsafe_allow_html=True,
        )
        col2.header("Brain Scan")
        col2.write("")
        col2.write("")

        col2.image(uploaded_file, use_column_width=True)
        progress_bar = col2.progress(0)

        for perc_completed in range(100):
            time.sleep(0.005)
            progress_bar.progress(perc_completed + 1)

        col2.write("")
        col2.info("MRI Scan was successfully uploaded!", icon="✔")

        col3.header("Tumor Class")

    with col3, col3.expander("Click Here to Classify MRI Scan!"):
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image = image.resize((255, 255))

            # Result of the interface
            segments = process(uploaded_file, backend)
            segments2 = process(uploaded_file, backend2)
            exp_image = Image.open(io.BytesIO(segments.content)).convert("RGB")
            result_string = segments2.text

            # st.image(exp_image, use_column_width=True)

            st.write("")
            st.write("")


            dirname1 = os.path.dirname(__file__)
            image_no = Image.open(dirname1 +'/OIPres.jpg')

            if result_string[1:3] == 'No':
                st.image(image_no, use_column_width=True)
                st.success(result_string)
            else:

               st.image(exp_image, use_column_width=True)
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
    st.title("Meet the Team")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # st.header("Aydogan")
        # st.write(os.getcwd())

        image = Image.open(dirname + "/aydogan.JPG")

        # Create container with centered image
        with st.container():
            st.markdown(
                "<h1 style='text-align: center'><a href='https://github.com/aydogan22'>Aydoğan</a></h1>",
                unsafe_allow_html=True,
            )

            col1.info("Researcher on Geospatial Analysis")
            # st.markdown('[Github](https://github.com/aydogan22)')
            st.image(image, use_column_width=True)
            # st.subheader('Researcher on Geospatial Analysis')

    with col2:
        image2 = Image.open(dirname + "/victor.jpg")

        # st.header("Victor")

        # st.image(image2, width=250)

        with st.container():
            st.markdown(
                "<h1 style='text-align: center'><a href='https://github.com/Victorvone'>Victor</a></h1>",
                unsafe_allow_html=True,
            )
            col2.info("Data-geek and Psychologist (M.Sc.)")
            # st.markdown('[Github](https://github.com/Victorvone)')
            st.image(image2, use_column_width=True)

    with col3:
        # st.header("Aurélien Biais")

        image3 = Image.open(dirname + "/aurelien.jpg")

        with st.container():
            st.markdown(
                "<h1 style='text-align: center'><a href='https://github.com/abiais'>Aurélien</a></h1>",
                unsafe_allow_html=True,
            )

            col3.info("Data Analytics Lead @recare")
            # st.markdown('[Github](https://github.com/abiais)')
            st.image(image3, use_column_width=True)

    with col4:
        # st.header("Aurélien Biais")
        image4 = Image.open(dirname + "/ivan.jpg")

        with st.container():
            st.markdown(
                "<h1 style='text-align: center'><a href='https://github.com/IvanAndjelkovic'>Ivan</a></h1>",
                unsafe_allow_html=True,
            )
            col4.info("Head of Controlling")
            # st.markdown('[Github](https://github.com/IvanAndjelkovic)')
            st.image(image4, use_column_width=True)
