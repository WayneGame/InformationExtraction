import csv
import config
from sklearn import preprocessing
import pandas as pd

def read_in_csv(csv_file):
    with open(csv_file, 'r', encoding='utf-8') as fp:
        reader = csv.reader(fp, delimiter=',', quotechar='"')
        data_read = [row for row in reader]
        
    return data_read

def get_data(filename):
    data = read_in_csv(filename)
    data_dict = {}
    for row in data[1:]:
        category = row[0]
        text = row[1]
        if (category not in data_dict.keys()):
            data_dict[category] = []
        data_dict[category].append(text)

    return data_dict  

def get_stopwords(path=config.STOPWORDS_PATH):
    stopwords = read_in_csv(path)
    stopwords = [word[0] for word in stopwords]
    stemmed_stopwords = [config.stemmer.stem(word) for word in stopwords]

    stopwords = stopwords + stemmed_stopwords
    return stopwords

def get_labels(labels):
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    return le

def create_dataset(data_dict, le, epoch):
    text = []
    labels = []
    for topic in data_dict:
        label = le.transform([topic])
        text = text + data_dict[topic]
        this_topic_labels = [label[0]]*len(data_dict[topic])
        labels = labels + this_topic_labels
    docs = {'text':text, 'label':labels}
    frame = pd.DataFrame(docs)
    return frame


def pad(a,i): 
    return a[0:i] if len(a) > i else a + [0] * (i-len(a))
