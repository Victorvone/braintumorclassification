from PIL import Image
from io import BytesIO
import numpy as np


input_shape = (255,255)


def read_image(image_encoded):
    pil_image = Image.open(BytesIO(image_encoded))
    return pil_image

def preprocess(image: Image.Image):
    image = image.resize(input_shape)
    image = np.asfarray(image)
    image = np.expand_dims(image,0)
    return image
