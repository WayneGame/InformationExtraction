import config
import sim_tfidf_utils as utils

import pandas as pd
import pickle
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

from tqdm import tqdm



model = load_model("../models/covid_model_2.h5")
tokenizer = "../models/covid_tokenizer_2.pickle"


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

def predict(url):

    info = utils.extract_information_from_url(url)

    if info is None:
        return None

    info = pd.DataFrame.from_records([info])

    
    X_example = transform_text(tokenizer, info) # im Original new_example => text_input
    label_array = model.predict(X_example)
    new_label = np.argmax(label_array, axis=-1)

    return new_label[0]



if __name__ == "__main__":
    interesting_webpages = []
    
    # "Alle" Test-Webseiten
    webpages_df = pd.read_csv("../input/webpages.csv")

    # Test, ob die Seite interessant ist:
    for _, row in tqdm(webpages_df.iterrows(), total=len(webpages_df)):
        label = predict(row["LINK"])
        if label == 1:
            interesting_webpages.append(row["LINK"]) # TODO: ggf müsste man hier schon nach LAND usw. unterscheiden? Hmm?!

    # Alle als interssant eingestuften Webseiten liegen nun in interesting_webpages
    # Diese werden seperat in einer Datei gespeichert, um die Vorhersage vom Pre Processing zu trennen.
    interesting_webpages_df = pd.DataFrame(interesting_webpages, columns=["LINK"])
    interesting_webpages_df.to_csv("../input/interesting_webpages.csv")

