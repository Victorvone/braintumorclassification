FROM --platform=linux/amd64 tensorflow/tensorflow:2.10.0

COPY braintumorclassification /braintumorclassification
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

CMD uvicorn braintumorclassification.api.fast:app --host 0.0.0.0 --port $PORT
