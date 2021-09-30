"""
Die Tabelle webpages.csv hat Links zu verschiedenen Webseiten gespeichert und 
dazu ein Label, ob diese Informationen zu Corona-Beschränlungen
enthält.
Hier wird der Link zum eigentlichen Webtext verarbeitet.
"""
import config
import re
import requests

import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from sklearn import model_selection

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
    #text = text.translate({ord(c): "" for c in config.SPECIAL_CHARACTERS_BERT}) #TODO: das Variabel machen
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


def run():
    # Laden der csv-Datenbank
    df = pd.read_csv("../input/webpages.csv", delimiter=",")

    data = []

    # Wandele den Link in ein bearbeiteten Dataframe
    for _, row in tqdm(df.iterrows(), total=len(df)):
        info  = extract_information_from_url(row["LINK"])
        if not info:
            continue

        # Nutzen wir die Impfeinträge?
        if config.USE_VACCINE:
            label = 0 if row["LABEL"] == "nein" else 1
        else:
            label = 0 if row["LABEL"] == "nein" else 1 if row["LABEL"] == "ja" else 2

        info["label"] = label
        info["kfold"] = -1
        data.append(info)

    processed_data = pd.DataFrame.from_records(data).fillna('none')
    del(df)

    # durchwürfeln der Daten
    processed_data = processed_data.sample(frac=1).reset_index(drop=True)
    
    # Label zum gleichmäßigem teilen
    y = processed_data.label.values

    # initiate the kfold class from the model_selection module
    kf = model_selection.StratifiedKFold(n_splits=config.K_FOLD_SPLITS)

    # füllen den kfold Spalte
    for f, (t_, v_) in enumerate(kf.split(X=processed_data, y=y)):
        processed_data.loc[v_, 'kfold'] = f
    
    # speichern der bearbeiteten Daten
    processed_data.to_csv("../input/web_pages_folds.csv", index=False)

if __name__ == "__main__":
    run()