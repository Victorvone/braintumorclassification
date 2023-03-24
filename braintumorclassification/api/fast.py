import sys
from fastapi import FastAPI, File, UploadFile
from braintumorclassification.ml_logic.predict_and_explain import predict_and_gradcam
import math
from braintumorclassification.ml_logic.registry import load_model

from braintumorclassification.api.prediction import read_image
from braintumorclassification.api.prediction import preprocess
from uvicorn import run
from starlette.responses import Response
import io
from PIL import Image

sys.path.append("../")
# run in main folder following command to start api:
# uvicorn userinterface.app:app --port 8080 --reload


# Fill in the classes
classes = ["glioma", "meningioma", "notumor", "pituitary"]


app = FastAPI()
app.state.model = load_model()
app.state.model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)


@app.post("/predict4")
async def predict_image(file: UploadFile = File(...)):
    # Read the file uploaded by user
    image = read_image(await file.read())
    # after doing preprocessing
    image = preprocess(image)

    # make prediction

    res_total = predict_and_gradcam(app.state.model, image)

    # getting a expl_image

    exp_image = res_total[1]
    r = Image.fromarray(exp_image, "RGB")

    bytes_io = io.BytesIO()
    r.save(bytes_io, format="PNG")

    return Response(bytes_io.getvalue(), media_type="image/png")


@app.post("/predict5")
async def predict_string(file: UploadFile = File(...)):
    # getting a string
    image = read_image(await file.read())
    # after doing preprocessing
    image = preprocess(image)
    # make prediction




    res_string = predict_and_gradcam(app.state.model, image)[0][0]

    max_value = res_string.max()
    # max_value = "{0:.0}%".format(max_value)
    max_position = res_string.argmax()
    round_max_value = math.floor(max_value*10000)/10000
    tumor_type = classes[max_position].capitalize()

    if max_position ==2:
        result_string = f'No tumor : {round_max_value:.2%} '
    else:
        result_string = f'{tumor_type} tumor : {round_max_value:.2%} '

    return result_string




@app.get("/")
def root():
    return {"greeting": "Hello"}


if __name__ == "__main__":
    run(app, debug=True)
