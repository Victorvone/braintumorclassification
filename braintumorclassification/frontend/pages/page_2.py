import streamlit as st
from PIL import Image


im = Image.open('/home/aydogan/code/Victorvone/braintumorclassification/braintumorclassification/frontend/brain.png')
st.set_page_config(layout="wide", page_title="Brain Tumor Classification and Explainability App", page_icon = im)

col1, col2, col3, col4= st.columns([6,6,6,6])

col1.header('Aydoğan Avcıoğlu, MSc')
image = Image.open('/home/aydogan/code/Victorvone/braintumorclassification/braintumorclassification/frontend/pages/aydogan.JPG')
col1.image(image, caption='')
col1.info('Phd Candidate in geoscience, working geospatial data science')

icon_size = 20

col1.button('Github', 'https://github.com/aydogan22', 'Aydogan Github Page', icon_size)

col2.header('Victor')
image2 = Image.open('/home/aydogan/code/Victorvone/braintumorclassification/braintumorclassification/frontend/pages/victor.JPG')
col2.image(image2, caption='')
col2.info('Add Info')

icon_size = 20

col2.button('Github', 'https://github.com/Victorvone', 'Victor Github Page', icon_size)

col3.header('Aurélien Biais')
image3 = Image.open('/home/aydogan/code/Victorvone/braintumorclassification/braintumorclassification/frontend/pages/aurelien.jpg')
col3.image(image3, caption='')
col3.info('Add Info')

icon_size = 20

col3.button('Github', 'https://github.com/abiais', 'Aurélien Github Page', icon_size)




col4.header('Ivan Andjelkovic')
image4 = Image.open('/home/aydogan/code/Victorvone/braintumorclassification/braintumorclassification/frontend/pages/ivan.JPG')
col4.image(image4, caption='')
col4.info('Add Info')

icon_size = 20

col4.button('Github', 'https://github.com/IvanAndjelkovic', 'Ivan Github Page', icon_size)
