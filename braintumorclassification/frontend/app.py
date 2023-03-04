import streamlit as st
import numpy as np
import pydeck as pdk
import requests
from io import StringIO
from io import BytesIO
from PIL import Image


st.markdown('''
First view of our app!
''')



uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg'])
if uploaded_file is not None:
    # To convert to a bytes IO:
    bytes_io = BytesIO(uploaded_file.getvalue())
    st.write(bytes_io)

    # Can be used wherever a "file-like" object is accepted:
    img = Image.open(bytes_io)
    st.image(img)
