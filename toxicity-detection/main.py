from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

file_path = '../dataset/train.csv'
df = pd.read_csv(file_path)

path = os.path.join('..', 'toxic-comment-detection','model.keras')
model = tf.keras.models.load_model(path)

tokenizer = Tokenizer(num_words=20000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['comment_text'])

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
def predict():
    text = request.form.get('textarea')
    if(text==""):
        return render_template("index.html", result="Enter text")
    result = predict_text(text)
    return render_template("index.html", result=result)



def preprocess_text(text):
    # Tokenize and pad the input text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')
    return padded_sequence

def predict_text(text):
    # Preprocess the text
    padded_sequence = preprocess_text(text)
    # Predict using the loaded model
    prediction = model.predict(padded_sequence)
    # Interpret the prediction (assuming binary classification: 0 = non-toxic, 1 = toxic)
    if prediction:
        if prediction[0][0] >= 0.5:
            return "Toxic speech detected!!"
        else:
            return "This text is non-toxic"
    else:
        return "error"




if __name__ == '__main__':
    app.run(debug=True)
   

