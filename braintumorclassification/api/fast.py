from fastapi import FastAPI

app = FastAPI()


@app.get("/predict")
def predict():
    """
    predict endpoint
    takes input from user and returns prediction
    """
    return None


@app.get("/")
def root():
    return {"greeting": "Hello"}
