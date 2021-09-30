import config

import sim_tfidf_config
import semantic_bert_config as bert_config
import re
import csv
import requests

from bs4 import BeautifulSoup

import tensorflow as tf
import numpy as np

"""
Da man in der deutschen Sprache Worte beliebig zusammenfügen kann,
zählt beispielsweise das Wort Coronakontaktbeschränkungen nicht zu der Termfrequency des Wortes Corona.
Deswegen, werden gefundene thematisch interessante Schlagworte einfach nochmal hinten an den Text angefügt.
"""
def add_keywords(text, word_list=sim_tfidf_config.WORD_LIST):
    text = str(text).lower()

    return_text = text
    for list_ in word_list: # config.
        for item in list_:
            return_text += text.count(item) * (" " + item)

    for key, list_ in sim_tfidf_config.THEMEN.items():
        for item in list_:
            return_text += text.count(item) * (" " + key)

    return return_text

def read_in_csv(csv_file):
    with open(csv_file, 'r', encoding='utf-8') as fp:
        reader = csv.reader(fp, delimiter=',', quotechar='"')
        data_read = [row[0] for row in reader]
    return data_read[1:] # das erste Wort ist der Spaltenname

def get_webpage(url):
    return requests.get(url, timeout=100).content

def clean_text(text):
    text = re.sub(config.RE_TAGS,"", text)
    text = re.sub(config.RE_COND_COMMENTS,"", text)
    text = re.sub(config.RE_LINK,"", text)

    # TODO: Ist das Entfernen der Zahlen wirklich eine richtige Entscheidung?
    #text = re.sub(r'\d', " ", text)

    text = re.sub(r'-', " ", text)
    #text = text.translate({ord(c): "" for c in config.SPECIAL_CHARACTERS})
    text = text.translate({ord(c): "" for c in config.SPECIAL_CHARACTERS_BERT})
    text = " ".join(text.split())
    return text.lower()

def get_tags(html):
    soup = BeautifulSoup(html, 'html.parser')
    text = ""

    for t in soup.findAll("meta"):
        if t.get("name") and t.get("content"):
            txt = t["content"]
            #if "=" not in txt and "/" not in txt:
            if "description" in t.attrs.get('name'):
                text += f"{txt} "

    return clean_text(" ".join(text.split()))

def get_h_elements(html):
    soup = BeautifulSoup(html, 'html.parser')
    text = ""

    for h in ["h1", "h2", "h3", "h4", "h5", "h6"]:
        for e in soup.findAll(h):
            text += f"{e.get_text()} "

    return clean_text(" ".join(text.split()))

def get_cleand_url(url):
    url = re.sub(r'https?:\/\/(www\.)?.+\.de\/', "", url)
    url = re.sub(r'-', " ", url)
    url = re.sub(r'\.html|\.php', " ", url)
    url = re.sub(r'\d', " ", url)
    #url = url.translate({ord(c): "" for c in config.SPECIAL_CHARACTERS})
    url = url.translate({ord(c): "" for c in config.SPECIAL_CHARACTERS_BERT})
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
TODO: Diese Funktion und alle dazugehörigen in eine util.py legen, da app.py, train.py und diese die Funktion nutzen!!!
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


    
# Create a custom data generator
class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    def __init__(
            self,
            sentence_pairs,
            labels,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            include_targets=True,
            tokenizer=None
        ):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets
        # Load our BERT Tokenizer to encode the text.
        # We will use base-base-uncased pretrained model.
        self.tokenizer = tokenizer

        self.indexes = np.arange(len(self.sentence_pairs))
        self.on_epoch_end()


    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.sentence_pairs) // self.batch_size

    def __getitem__(self, idx):
        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        sentence_pairs = self.sentence_pairs[indexes]

        # With BERT tokenizer's batch_encode_plus batch of both the sentences are
        # encoded together and separated by [SEP] token.
        encoded = self.tokenizer.batch_encode_plus(
            sentence_pairs.tolist(),
            add_special_tokens=True,
            max_length=bert_config.MAX_LENGTH,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="tf",
            truncation=True,
            padding="max_length"
        )

        # Convert batch of encoded features to numpy array.
        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

        # Set to true if data generator is used for training/validation.
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [input_ids, attention_masks, token_type_ids], labels
        else:
            return [input_ids, attention_masks, token_type_ids]

    def on_epoch_end(self):
        # Shuffle indexes after each epoch if shuffle is set to True.
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)
