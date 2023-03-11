# run in main folder following command to start api:
# uvicorn userinterface.app:app --port 8080 --reload

from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
import numpy as np
from prediction import read_image
from prediction import preprocess

#Fill in the classes
classes=['glioma', 'meningioma', 'notumor', 'pituitary']
# Fill in the Model
filename="/home/ivana/code/Victorvone/braintumorclassification/models/EfficientNetv2.h5" # "home/ivana/code/Victorvone/braintumorclassification/models/EfficientNetv2.h5" for binary case

app = FastAPI()



@app.post("/predict3")
async def predict(file: UploadFile = File(...)):
    #Read the file uploaded by user
    image = read_image(await file.read())
    #after doing preprocessing
    image = preprocess(image)
    #make prediction
    app.state.model = load_model(filename, compile=False)
    app.state.model.compile(loss = 'categorical_crossentropy',
                  optimizer = "adam",
                  metrics = ['accuracy'])
    res = app.state.model.predict(image) # Predict prediction
    val0 = res[0][0]
    val1 = res[0][1]
    val2 = res[0][2]
    val3 = res[0][3]
    return f'''
    The result is {val0} for {classes[0]}/n
    The result is {val1} for {classes[1]}/n
    The result is {val2} for {classes[2]}/n
    The result is {val3} for {classes[3]}.
    '''
    # return image.shape



@app.get("/")
def root():
    return {"greeting": "Hello"}


if __name__ == "__main__":
    uvicorn.run(app, debug=True)
