from flask import Flask, render_template, request, jsonify
from model_config import *
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


import pickle

app = Flask(__name__)

# Load Tokenizer
with open('./static/models/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load Model
model = load_model("./static/models/model.h5")


def decode_sentiment(score, include_neutral=True):
    if include_neutral:
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NOT_RECOMMENDED
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = RECOMMENDED

        return label
    else:
          return NOT_RECOMMENDED if score < 0.5 else RECOMMENDED

@app.route("/predict", methods=["POST"])
def predict():

    text = request.data.decode('unicode-escape')
    print(text)

    include_neutral = True
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score, include_neutral=include_neutral)

    return jsonify({"label": label, "score": float(score)})

# Home route for the home page
@app.route("/review.html")
def review():

    return render_template("review.html")

@app.route("/")
def home():

    return render_template("Index.html")

@app.route("/Index.html")
def index():

    return render_template("Index.html")

@app.route("/TopStreamers.html")
def streamers():

    return render_template("TopStreamers.html")

@app.route("/Games.html")
def games():

    return render_template("Games.html")

@app.route("/ML.html")
def ML():

    return render_template("ML.html")






if __name__ == "__main__":

    app.run()