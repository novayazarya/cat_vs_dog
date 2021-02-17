import io
import numpy as np
import tensorflow as tf
import os, base64, json, requests
from flask import Flask, render_template, request, flash
from PIL import Image

from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

model = load_model('/home/enzo/cat_vs_dog/CatDog.h5', compile=False)
graph = tf.Graph()

app = Flask(__name__)
app.secret_key = os.urandom(24)

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":
        # Display the image that was uploaded
        image = request.files["file"]
        uri = "data:;base64," + base64.b64encode(image.read()).decode("utf-8")
        image.seek(0)

        flash(predict(image.read()))

    else:
        # Display a placeholder image
        uri = "/static/placeholder.png"

    return render_template("index.html", image_uri=uri)

def load_image(img_bytes: bytes, size: list =[200,200]) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert('RGB')
    img = img.resize(size, Image.NEAREST)
    img = img_to_array(img)
    return img.reshape(1, *size, 3)

def predict(img: np.ndarray) -> str:
    catdog_classifier = model
    #with graph.as_default():
    answer = catdog_classifier.predict(load_image(img))

    return 'cat: {:.1f}% \ndog: {:.1f}%'.format(*tuple(answer[0] * 100))
