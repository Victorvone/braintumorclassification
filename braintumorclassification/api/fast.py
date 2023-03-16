# run in main folder following command to start api:
# uvicorn userinterface.app:app --port 8080 --reload

from fastapi import FastAPI, File, UploadFile
from braintumorclassification.ml_logic.registry import load_model

# import numpy as np
from braintumorclassification.api.prediction import read_image
from braintumorclassification.api.prediction import preprocess
from uvicorn import run

# Fill in the classes
classes = ["glioma", "meningioma", "notumor", "pituitary"]
# Fill in the Model
filename = "models/EfficientNetv2.h5"

app = FastAPI()
app.state.model = load_model()


@app.get("/predict3")
async def predict(file: UploadFile = File(...)):
    # Read the file uploaded by user
    image = read_image(await file.read())
    # after doing preprocessing
    image = preprocess(image)
    # make prediction
    app.state.model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    res = app.state.model.predict(image)  # Predict prediction
    val0 = res[0][0]
    val1 = res[0][1]
    val2 = res[0][2]
    val3 = res[0][3]
    return f"""
    The result is {val0} for {classes[0]}/n
    The result is {val1} for {classes[1]}/n
    The result is {val2} for {classes[2]}/n
    The result is {val3} for {classes[3]}.
    """
    # return image.shape


@app.get("/")
def root():
    return {"greeting": "Hello"}


if __name__ == "__main__":
    run(app, debug=True)
