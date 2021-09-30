import config

import pandas as pd
import pickle
import numpy as np

from keras.preprocessing.sequence import pad_sequences

from keras.models import load_model

import config
import re
import requests

from bs4 import BeautifulSoup

import time
import json
from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
#app.config['CORS_HEADERS'] = 'Content-Type'

#df = create_dataset(data_dict, le)

# train_model(df, le)
# load_and_evaluate_existing_model(config.MODEL_PATH, config.TOKENIZER_PATH, df, le)

model = load_model("../models/covid_model_2.h5")
tokenizer = "../models/covid_tokenizer_2.pickle"


def get_webpage(url):
    return requests.get(url, timeout=100).content

def clean_text(text):
    text = re.sub(config.RE_TAGS,"", text)
    text = re.sub(config.RE_COND_COMMENTS,"", text)
    text = re.sub(config.RE_LINK,"", text)

    # TODO: Ist das Entfernen der Zahlen wirklich eine richtige Entscheidung?
    text = re.sub(r'\d', " ", text)

    text = re.sub(r'-', " ", text)
    text = text.translate({ord(c): "" for c in config.SPECIAL_CHARACTERS})
    text = " ".join(text.split())
    return text.lower()

def get_tags(html):
    soup = BeautifulSoup(html, 'html.parser')
    text = ""

    for t in soup.findAll("meta"):
        if t.get("name"):
            txt = t["content"]
            #if "=" not in txt and "/" not in txt:
            if "description" in t.attrs.get('name'):
                text += f"{txt} "

    return clean_text(" ".join(text.split()))

def get_cleand_url(url):
    url = re.sub(r'https?:\/\/(www\.)?.+\.de\/', "", url)
    url = re.sub(r'-', " ", url)
    url = re.sub(r'\.html|\.php', " ", url)
    url = re.sub(r'\d', " ", url)
    url = url.translate({ord(c): "" for c in config.SPECIAL_CHARACTERS})
    url = url.lower()
    return " ".join(url.split("/"))

def get_webpage_title(html):

    soup = BeautifulSoup(html, 'html.parser')
    text = ""

    for t in soup.find("title"):
        text += f"{t} "

    return clean_text(" ".join(text.split()))


def get_cleaned_webpage_text(html):
    text = ""
    soup = BeautifulSoup(html, 'html.parser')


    for t in soup.find_all(text=True):
        if t.parent.name not in config.TAG_BLACKLIST:
            text += f"{t} "

    text = " ".join(text.split())
    return clean_text(text)


def get_word_count(text, word_list=config.COVID_WORDS):
    text = str(text).lower()

    count = 0
    for item in word_list:
        count += text.count(item)
    
    return count

"""
extrahiert die interessanten und gesäuberten
Informationen einer Webseite zurück.
Transformiert eine Webseite in ein Dictionary.
"""
def extract_information_from_url(url):
    try:
        html = get_webpage(url)
    except:
        return None

    link = get_cleand_url(url)
    body = get_cleaned_webpage_text(html)
    meta = get_tags(html)
    title = get_webpage_title(html)

    covid_word_count     = get_word_count(body, word_list=config.COVID_WORDS)
    covid_word_count_url = get_word_count(link, word_list=config.COVID_WORDS)
    restriction_word_count     = get_word_count(body, word_list=config.RESTRICTION_WORDS)
    restriction_word_count_url = get_word_count(link, word_list=config.RESTRICTION_WORDS)

    return {
        "link": link,
        "body": body,
        "meta_data": meta,
        "title": title,
        "covid_word_count": covid_word_count,
        "covid_word_count_url": covid_word_count_url,
        "restriction_word_count": restriction_word_count,
        "restriction_word_count_url": restriction_word_count_url,
    }


def transform_text(tokenizer, df):
    if (isinstance(tokenizer, str)):
        tokenizer = load_tokenizer(tokenizer)

    # Tokenizing der Link Informationen
    X_input = tokenizer.texts_to_sequences(df['link'].values)
    X_input = pad_sequences(X_input, maxlen=config.MAX_LINK_SEQUENCE_LENGTH)
    # Tokenizing der Meta Informationen
    X_meta = tokenizer.texts_to_sequences(df['meta_data'].values)
    X_meta = pad_sequences(X_meta, maxlen=config.MAX_META_SEQUENCE_LENGTH)
    # Tokenizing der Titel Informationen
    X_title = tokenizer.texts_to_sequences(df['title'].values)
    X_title = pad_sequences(X_title, maxlen=config.MAX_TITLE_SEQUENCE_LENGTH)
    # Tokenizing des Seiteninhalts
    X_body = tokenizer.texts_to_sequences(df['body'].values)
    X_body = pad_sequences(X_body, maxlen=config.MAX_BODY_SEQUENCE_LENGTH)
    covid_word_count = df['covid_word_count'].values
    covid_word_count_url = df['covid_word_count_url'].values
    restriction_word_count = df['restriction_word_count'].values
    restriction_word_count_url = df['restriction_word_count_url'].values

    X_input = np.concatenate([X_input, X_meta], axis=-1)
    X_input = np.concatenate([X_input, X_title], axis=-1)
    X_input = np.concatenate([X_input, X_body], axis=-1)

    covid_word_count = np.expand_dims(covid_word_count, axis=(-1))
    X_input = np.concatenate([X_input, covid_word_count], axis=-1)

    covid_word_count_url = np.expand_dims(covid_word_count_url, axis=(-1))
    X_input = np.concatenate([X_input, covid_word_count_url], axis=-1)

    restriction_word_count = np.expand_dims(restriction_word_count, axis=(-1))
    X_input = np.concatenate([X_input, restriction_word_count], axis=-1)

    restriction_word_count_url = np.expand_dims(restriction_word_count_url, axis=(-1))
    X_input = np.concatenate([X_input, restriction_word_count_url], axis=-1) # Schlussendlich alles zusammefügen

    return X_input

def load_tokenizer(filename):
    with open(filename, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer 


def test_new_example(model, tokenizer, le, text_input):
    X_example = transform_text(tokenizer, [text_input]) # im Original new_example => text_input
    label_array = model.predict(X_example)
    new_label = np.argmax(label_array, axis=-1)
    print(new_label)
    print(le.inverse_transform(new_label))

@app.route("/predict", methods=['POST'])
@cross_origin()
def predict():
    sentence_object = request.json
    url = sentence_object['text']

    print(url)
    info = extract_information_from_url(url)
    print(info)
    info = pd.DataFrame.from_records([info])

    start_time = time.time()
    
    X_example = transform_text(tokenizer, info) # im Original new_example => text_input
    label_array = model.predict(X_example)
    new_label = np.argmax(label_array, axis=-1)

    response = {}
    response["response"] = {
        "label": str(new_label[0]),
        "label_": "Ja, ist interessant" if new_label[0] == 1 else "Uninteressant", # Normalerweise mit Labelencoder, ist bei binär allerdings uninteressant
        "pred": str(label_array[0][new_label][0]),
        "time_taken": str(time.time() - start_time),
    }
    response = app.response_class(
        response=json.dumps(response),
        status=200,
        mimetype='application/json'
    )

    #resp = flask.Response(flask.jsonify(response))
    #response.headers['Access-Control-Allow-Origin'] = '*' # Nicht nötig mit flask_cors, führt zu Fehler wegen multiple_headers!!!!
    return response


if __name__ == "__main__":

    app.run(host="localhost", port="9999")
