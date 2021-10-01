import pandas as pd
from sentence_transformers import util
import torch
import sim_tfidf_config as config
import sim_tfidf_utils  as utils
import semantic_bert_config as bert_config
import tensorflow as tf
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

import numpy as np
import time
import transformers

import re

from tqdm import tqdm

# bm25 wird in der init Funktion initialisiert, da sich sein Corpus je nach
# Selector ändern kann
bm25 = None
corpus = None
bert = None
tfidf_vectorizer = None
tfidf_matrix     = None
bert_encoded = None
tokenizer = None
nlp = None
bert_corpus_embeddings = []
spacy_vectors = []

# Bert labels #TODO
labels = ["komplett anders", "ähnlich", "schon neutral"]

test_questions = pd.read_csv(config.TEST_PATH)


"""
Der Bert-Schlater dient nur dem ausgewähltem Datensatz.
Für sentenceBert ist es wichtig, dass die Sätze noch ihre Satzzeichen enthalten.
Steht der Schalter auf "all", werden alle Einträge auch die "irrelevanten" verarbeitet
Im Normalfall werden nur die intrerssanten verarbeitet.
"""
def init(bert=None):
    global corpus
    
    if bert == "bert":
        corpus = pd.read_csv(f"{config.CORPUS_SAVE_PATH}_bert").sample(frac=1) # TODO, das ist für Sentence Bert
    elif bert == "all":
        corpus = pd.read_csv(f"{config.CORPUS_SAVE_PATH}_all") # TODO "all" weg und nur config.CORP...
    else:
        corpus = pd.read_csv(config.CORPUS_SAVE_PATH) # TODO: normal
    
    
    initBM25()
    initTFIDF()
    initSpaCy()
    #initBERT()

# #####################################################
# BERT Funktionen
# #####################################################
def initBERT():
    global bert
    global tokenizer
    global bert_corpus_embeddings
    global corpus

    max_length = 128  # Maximum length of input sentence to the model.
    batch_size = 32
    epochs = 2

    tokenizer = transformers.BertTokenizer.from_pretrained(
            bert_config.BERT_MODEL, do_lower_case=True
        )

    bert = tf.keras.models.load_model("../models/sim_bert_model_2epochs.bin")

    """
    Berechnen und Speicher der Embeddings der Corpussätze in eine Liste aus Listen von Dicts, der Gestalt
    der Index entspricht dem Index des Corpus -> nicht nötig die url zu speichern. 
    [[{
    input_ids: xx
    attention_mask: xx
    token_type_ids: xx
    }, ...],
    ...
    ]

    Anschließendes iterieren über diese Liste und vergleich mit dem Embedding der Query mit den Listeneinträgen
    und speichern der Summer der besten k Scores / k  samt dem Index in einer weiteren Liste -> 
    """
    # Ausdruck zum Teilen von Sätzen. So bleibt beispielsweise U.S. in einem Satz oder 1. 
    sentence_splitter = re.compile('(?<!\d\.)(?<!\w\.\w.)(?<=\.|\?|\:).{0,1}\s\W{0,1}')
    for idx, row in tqdm(corpus.iterrows(), total=len(corpus)):
        embedded_sentences = []
        # Alle Sätze in einer Liste
        corpus_sentences = sentence_splitter.split(row["corpus"])
        for s in corpus_sentences:
            embedded_sentences.append(calc_embedding(s))

        bert_corpus_embeddings.append((idx, embedded_sentences))


def calc_embedding(sentence):
    return tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=128,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=False,
            return_tensors="tf",
            truncation=True
        )


def check_similarity(sentence1, sentence2):
    global bert
    global labels

    sentence_pairs = np.array([[str(sentence1), str(sentence2)]])

    test_data = utils.BertSemanticDataGenerator(
        sentence_pairs, labels=labels, batch_size=1, shuffle=False, include_targets=False, tokenizer=tokenizer # labels = None
    )

    proba = bert.predict(test_data)[0]
    idx = np.argmax(proba)
    proba = f"{proba[idx]: .2f}%"
    pred = labels[idx]
    return idx, proba

"""
sentence1und2 sind dicts
{
    "input_ids": xx
    "attention_mask": xx
    "token_type_ids": xx
}
"""
# Test, ob es sinn macht die Embeddings vorher zu berechnen und sie dann nur zu verbinden
def combine_sentences_for_bert(sentence1, sentence2):
    max_length = 128

    # Absschneiden des [CLS]-Tokens und anschließendes zusammenfügen der beiden Sätze
    input_ids_satz2 = sentence2["input_ids"][:, 1:]
    attention_mask_satz2 = sentence2["attention_mask"][:, 1:]
    token_type_ids_satz2 = sentence2["token_type_ids"][:, 1:]

    # Zusammenfügen der Beiden  attention_masks, token_type_ids
    input_ids = tf.concat([sentence1["input_ids"], input_ids_satz2], axis = 1)
    attention_masks = tf.concat([sentence1["attention_mask"], attention_mask_satz2], axis = 1)

    # Die TokenType Ids müssen noch von 0 auf 1 gebracht werden, für die Maske des zweiten Satzes
    token_type_ids = tf.add(token_type_ids_satz2, 1)
    token_type_ids = tf.concat([sentence1["token_type_ids"], token_type_ids], axis = 1)

    # Padding 
    input_ids = tf.keras.preprocessing.sequence.pad_sequences(
        input_ids, padding="post", maxlen=max_length, value=0)
    token_type_ids = tf.keras.preprocessing.sequence.pad_sequences(
        token_type_ids, padding="post", maxlen=max_length, value=0)
    attention_masks = tf.keras.preprocessing.sequence.pad_sequences(
        attention_masks, padding="post", maxlen=max_length, value=0)

    # TODO: so sollten die nicht gespeichert sine
    input_ids = np.array(input_ids, dtype="int32")
    attention_masks = np.array(attention_masks, dtype="int32")
    token_type_ids = np.array(token_type_ids, dtype="int32")

    return [input_ids, attention_masks, token_type_ids]

def getTopNBERTArticles_entailment(query, n, n_sentences, bm25=True, rerank=10):
    global corpus
    global bert_corpus_embeddings
    global bert

    webpage_ratings = []

    sentence_embeddings = []
    indices = getTopNbm25Indices(query, rerank) if bm25 else getTopNTFIDFIndices(query, rerank)

    for i in indices:
        sentence_embeddings.append(bert_corpus_embeddings[i])


    print("calculate sentence similarities – Der Mist braucht eeeeewing!!!")
    for idx, embedded_sentences in tqdm(sentence_embeddings, total=len(sentence_embeddings)):
        best_sentences = []
        query_embedding = calc_embedding(query)

        # TODO noch entfernen
        if len(embedded_sentences) > 25:
            embedded_sentences = embedded_sentences[:25]

        for j, sentence_embedding in enumerate(embedded_sentences):
            test_data = combine_sentences_for_bert(query_embedding, sentence_embedding)

            proba = bert.predict(test_data)[0]
            proba_idx = np.argmax(proba)
            
            # proba_idx = 0 -> contradiction
            # proba_idx = 1 -> entailment
            # proba_idx = 2 -> neutral
            best_sentences.append(proba[1])


        # Die Wahrscheinlichkeiten für alle ähnlichen Sätze 
        # sind nun in best_sentences. 
        top_n_scores_sum = sum(sorted(best_sentences, key=lambda i: i, reverse=True)[:n_sentences])
        webpage_ratings.append({
            "idx": idx,
            "score": top_n_scores_sum
        })

    if n < 0:
        n = len(embedded_sentences)

    top_n_indices = sorted(webpage_ratings, key=lambda i: i["score"], reverse=True)[:n]
    top_n_indices = [item["idx"] for item in top_n_indices]

    return corpus.iloc[top_n_indices]["url"].to_list()




# TODO: Die ganze Sache ließe sich erheblich beschleunigen, wenn man die embeddings des Corpus vorher berechnen würde
# Update, selbst dann nicht wirklich :(
def getTopNBERTArticles(query, n, clean=True):
    global corpus

    # Anpassen der Abfrage:
    if clean:
        query = utils.clean_text(str(query))
        query = utils.add_keywords(query)

    if n < 0:
        n = len(corpus)

    similarities = []

    for idx, row in corpus.iterrows():
        similarity, probability = check_similarity(query, row["corpus"])
        
        # 1 bedeutet, dass die Sätze sich ähnlich sind. "entailment"
        # 0 = contradiction
        # 2 = neutral
        if similarity == 1:
            similarities.append((idx, probability))


    #top_n_indices = sorted(range(len(similarities)), key=lambda i: similarities[i][1], reverse=True)[:n]
    top_n_indices = sorted(similarities, key=lambda i: i[1], reverse=True)[:n]
    top_n_indices = [item[0] for item in top_n_indices]

    return corpus.iloc[top_n_indices]["url"].to_list()
    
# #####################################################
# BM25 Funktionen
# #####################################################
def initBM25():
    global bm25
    global corpus

    # Initialisieren von BM25
    #bm25 = BM25Okapi(corpus["corpus"].tolist())
    bm25 = BM25Okapi([doc.split(" ") for doc in corpus["corpus"].tolist()])


def getTopNbm25Indices(query, n):
    global corpus
    global bm25
    global stopwords

    if n <= 0:
        n = len(corpus)

    # Anpassen der Abfrage:
    query = utils.clean_text(str(query))
    query = utils.add_keywords(query)

    return bm25.get_top_n(query.split(" "), corpus.index.values.tolist(), n=n)

def getTopNbm25Articles(query, n):
    global corpus
    global bm25
    global stopwords

    if n <= 0:
        n = len(corpus)

    # Anpassen der Abfrage:
    query = utils.clean_text(str(query))
    query = utils.add_keywords(query)

    return bm25.get_top_n(query.split(" "), corpus["url"].tolist(), n=n)

# #####################################################
# SpaCy Funktionen
# #####################################################
def initSpaCy():
    global nlp
    global spacy_vectors
    
    start = time.process_time() 

    nlp = spacy.load('de_core_news_lg')
    for _, row in corpus.iterrows():
        nina = nlp(row["corpus"])
        spacy_vectors.append({"vector": nina.vector, "norm_vector": nina.vector_norm})

    print("Spacy preprocess Time:")
    print(time.process_time() - start)

def spacy_sim(query, other):
    # So errechnet SpaCy intern die ähnlichkeit zwischen zwei nlp-Elementen
    # query ist hier ein nlp-Objekt und other ein dict  
    return np.dot(query.vector, other["vector"]) / (query.vector_norm * other["norm_vector"])

def getTopNSpaCyArticles(query, n=-1, clean=False):
    global nlp
    global spacy_vectors

    # Anpassen der Abfrage:
    if clean:
        query = utils.clean_text(str(query))
        query = utils.add_keywords(query)

    query = nlp(query)

    if n < 0:
        n = len(spacy_vectors)

    similarities = []


    for idx, nina in enumerate(spacy_vectors):
        similarities.append((idx, spacy_sim(query, nina)))


    #top_n_indices = sorted(range(len(similarities)), key=lambda i: similarities[i][1], reverse=True)[:n]
    top_n_indices = sorted(similarities, key=lambda i: i[1], reverse=True)[:n]
    top_n_indices = [item[0] for item in top_n_indices]

    return corpus.iloc[top_n_indices]["url"].to_list()
    
    

# #####################################################
# Benchmark Funktionen
# #####################################################
def calc_mAPkScore(k=5, algo="BM25", n_sentences=5, rerank=10, rerank_bm25=True):
    global corpus
    global test_questions

    if k <= 0:
        k = len(corpus)

    totalDocCount = 0
    AP = 0
    # Todo tina entfernen
    for tina, question in tqdm(test_questions.iterrows(), total=len(test_questions)):
        TPseen = 0
        TPtotal = 0
        APtmp = 0

        if algo == "tfidf":
            urls = getTopNTFIDFArticles(query=question["question"], n=-1)
        elif algo == "BM25":
            urls = getTopNbm25Articles(query=question["question"], n=-1)
        elif algo == "spacy":
            urls = getTopNSpaCyArticles(query=question["question"], n=-1)
        else: #BERT
            urls = getTopNBERTArticles_entailment(query=question["question"], n=-1, n_sentences=n_sentences, rerank=rerank, bm25=rerank_bm25)



        # jetzt wird geschaut, an welcher Stelle der gesuchte Artikel ist.
        for j, url in enumerate(urls):
            # urls sind die Links zu den passendsten Artikeln.
            if question["favoured_url"].count(url) > 0:
                #print('Gefunden an der Stelle :', j)
                #TPtotal = TPtotal + 1
                if j < k:
                    TPseen = TPseen + 1
                    APtmp = APtmp + TPseen / (j+1)

        # Ich kann wegen des Testaufbaus davon ausgehen, dass TPtotal immer 1 ist.
        TPtotal = 1

        AP = AP + APtmp/min(k, TPtotal)
        totalDocCount = totalDocCount + 1
        #break
    return AP / totalDocCount

def calc_MRRScore(algo="BM25", n_sentences=5, rerank_bm25=True, rerank=10):
    global corpus
    global test_questions

    counter = 0
    sum = 0
    for tina, question in tqdm(test_questions.iterrows(), total=len(test_questions)):
        
        if algo == "tfidf":
            urls = getTopNTFIDFArticles(query=question["question"], n=-1)
        elif algo == "BM25":
            urls = getTopNbm25Articles(query=question["question"], n=-1)
        elif algo == "spacy":
            urls = getTopNSpaCyArticles(query=question["question"], n=-1)
        else: #BERT
            urls = getTopNBERTArticles_entailment(query=question["question"], n=-1, n_sentences=n_sentences, rerank=rerank, bm25=rerank_bm25)


        # jetzt wird geschaut, an welcher Stelle der gesuchte Artikel ist.
        counter = counter + 1
        for j, url in enumerate(urls):
            # urls sind die Links zu den passendsten Artikeln.
            if question["favoured_url"].count(url) > 0:
                #print('Gefunden an der Stelle :', j)
                sum = sum + 1/(j+1)
                break

    return sum/counter

def calc_HasPositive(k=5, algo="BM25", n_sentences=5, rerank=10, rerank_bm25=True):
    global corpus
    global test_questions

    counter = 0
    sum = 0
    for tina, question in tqdm(test_questions.iterrows(), total=len(test_questions)):
        
        if algo == "tfidf":
            urls = getTopNTFIDFArticles(query=question["question"], n=-1)
        elif algo == "BM25":
            urls = getTopNbm25Articles(query=question["question"], n=-1)
        elif algo == "spacy":
            urls = getTopNSpaCyArticles(query=question["question"], n=-1)
        else: #BERT
            urls = getTopNBERTArticles_entailment(query=question["question"], n=-1, n_sentences=n_sentences, rerank=rerank, bm25=rerank_bm25)

        counter = counter + 1
        # jetzt wird geschaut, an welcher Stelle der gesuchte Artikel ist.
        for j, url in enumerate(urls):
            # ids ist ein Tupel (ID, TrueNews Sub-ID)
            if question["favoured_url"].count(url) > 0:
                #print('Gefunden an der Stelle :', j)
                sum = sum + 1/(j+1)
                break

            if j >= k:
                break

    return sum/counter

# #####################################################
# TF/IDF Funktionen
# #####################################################
"""
In der Init Funktion kann man zwischen Count und IDF Vectorizer wählen.
"""
def initTFIDF(count_vectorizer=False):
    global corpus
    global tfidf_vectorizer
    global tfidf_matrix

    # Initialisieren von BM25
    if count_vectorizer:
        tfidf_vectorizer = CountVectorizer(stop_words=utils.read_in_csv("../input/stopwords_ger.csv"))
    else:
        tfidf_vectorizer = TfidfVectorizer(stop_words=utils.read_in_csv("../input/stopwords_ger.csv"))

    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus["corpus"].tolist())

def getTopNTFIDFArticles(query, n):
    global tfidf_matrix
    global tfidf_vectorizer
    global corpus

    if n <= 0:
        n = len(corpus)
 

    query = utils.clean_text(str(query))
    query = utils.add_keywords(query)

    query_encoding = tfidf_vectorizer.transform([query])

    cos_scores = torch.from_numpy(cosine_similarity(query_encoding, tfidf_matrix))[0] # [0]
    cos_scores = cos_scores.cpu()

    #We use torch.topk to find the highest 5 scores
    top_results = torch.topk(cos_scores, k=n)

    urls = []

    for score, idx in zip(top_results[0], top_results[1]):
        urls.append(corpus.iloc[idx.numpy()]["url"])
        #print(corpus.iloc[idx.numpy()]["url"], "(Score: %.4f)" % (score))

    return urls

def getTopNTFIDFIndices(query, n):
    global tfidf_matrix
    global tfidf_vectorizer
    global corpus

    if n <= 0:
        n = len(corpus)
 

    query = utils.clean_text(str(query))
    query = utils.add_keywords(query)

    query_encoding = tfidf_vectorizer.transform([query])

    cos_scores = torch.from_numpy(cosine_similarity(query_encoding, tfidf_matrix))[0] # [0]
    cos_scores = cos_scores.cpu()

    #We use torch.topk to find the highest 5 scores
    top_results = torch.topk(cos_scores, k=n)

    urls = []

    for score, idx in zip(top_results[0], top_results[1]):
        urls.append(idx)
        #print(corpus.iloc[idx.numpy()]["url"], "(Score: %.4f)" % (score))

    return urls


def calcBenchmarks2():
    mapK = [1, 3, 5, 10, 20, -1]
    print("Calculating map@k Score:")
    for k in mapK:
        print(f"BM25  – mAP@{k} Score: {calc_mAPkScore(k=k, algo='BM25')}")
        print(f"TFIDF – mAP@{k} Score: {calc_mAPkScore(k=k, algo='tfidf')}")
        print(f"spaCy – mAP@{k} Score: {calc_mAPkScore(k=k, algo='spacy')}")

    hasPos = [1, 3, 5, 10, 20, 50]
    print("Calculating Has Pos Score:")
    for k in hasPos:
        print(f"BM25  – Has Pos @{k} Score: {calc_HasPositive(k=k, algo='BM25')}")
        print(f"TFIDF – Has Pos @{k} Score: {calc_HasPositive(k=k, algo='tfidf')}")
        print(f"spaCy – Has Pos @{k} Score: {calc_HasPositive(k=k, algo='spacy')}")

    print("Calculating MRR Score:")
    print(f"BM25  – Has MRR Score: {calc_MRRScore(algo='BM25')}")
    print(f"TFIDF – Has MRR Score: {calc_MRRScore(algo='tfidf')}")
    print(f"spaCy – Has MRR Score: {calc_MRRScore(algo='spacy')}")


def calcBenchmarks():
    for rerank in [10, 20, 50]:
        for n_sentences in [3,4,5,6]:
            mapK = [1, 3, 5, 10, 20, -1]
            print("Calculating map@k Score:")
            for k in mapK:
                #print(f"BM25  – mAP@{k} Score: {calc_mAPkScore(k=k, algo='BM25')}")
                #print(f"TFIDF – mAP@{k} Score: {calc_mAPkScore(k=k, algo='tfidf')}")
                #print(f"spaCy – mAP@{k} Score: {calc_mAPkScore(k=k, algo='spacy')}")
                print(f"BERT Rerank bm25 – mAP@{k} Score: {calc_mAPkScore(k=k, algo='bert', n_sentences=n_sentences, rerank=rerank, rerank_bm25=True)}")

            hasPos = [1, 3, 5, 10, 20, 50]
            print("Calculating Has Pos Score:")
            for k in hasPos:
                #print(f"BM25  – Has Pos @{k} Score: {calc_HasPositive(k=k, algo='BM25')}")
                #print(f"TFIDF – Has Pos @{k} Score: {calc_HasPositive(k=k, algo='tfidf')}")
                #print(f"spaCy – Has Pos @{k} Score: {calc_HasPositive(k=k, algo='spacy')}")
                print(f"BERT Rerank bm25 – Has Pos@{k} Score: {calc_HasPositive(k=k, algo='bert', n_sentences=n_sentences, rerank=rerank, rerank_bm25=True)}")

            #print("Calculating MRR Score:")
            #print(f"BM25  – Has MRR Score: {calc_MRRScore(algo='BM25')}")
            #print(f"TFIDF – Has MRR Score: {calc_MRRScore(algo='tfidf')}")
            #print(f"spaCy – Has MRR Score: {calc_MRRScore(algo='spacy')}")
            print(f"BERT Rerank bm25 – MRR@{k} Score: {calc_MRRScore(k=k, algo='bert', n_sentences=n_sentences, rerank=rerank, rerank_bm25=True)}")

if __name__ == "__main__":
    #init(bert="bert")
    # Dauert eeeeeeeeeewig, nicht plausibel
    #calcBenchmarks()
    
    init()
    calcBenchmarks2()