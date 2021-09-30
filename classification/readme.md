LSTM
---
## config.py

Allgemeine Einstellung, wie Datenpfade, Epochen, max_sequence_lenget, ...

---
## pre_process.py

Alle Links aus dem Link-Datensatz in die passende Repräsentation wandeln

---

## train.py

Trainiert das LSTM-Modell

---

## app.py

Ist ein unsauberer Server, um das LSTM-Modell zu testen.
Aufrufbar ist der Test-Server über ein kleine unsaubere Test-html seite 


BERT
---

## semantic_bert_config.py

Konfiguration für das BERT- Fine-Tuning

## semantic_bert_pre_process.py

Kombiniert die Beispielsätze aus der Datenbank für das Fine-Tuning

## semantic_bert_fine_tune.py

Übernimmt das eigentliche Fine-Tuining und speichern des Modells

BM25 und TFIDF
---

## im_tfidf_config.py

Einstellungen für TF/IDF und BM25

## sim_tfidf_utils.py

Hilfsfunkionen für BM25 und TFIDF Algorithmen

## sim_tfidf_get_interesting_pages.py

Geht alle Webseiten aus der Datenbank durch und speichert alle als relevant eingestuften.

## sim_tfidf_corpus_erstellen.py

Erstellt aus allen Seiten einen Document-Corpus für BM25 und TF/IDF

Benchmark Dateien
---

## sim_tfidf_calc_benchmarks.py
Errechnet die Benchmarks. Alles, was mit BERT zu tun hat ist uuuuuuunglaublich Rechenintensiv und zeitaufwendig.

SVM Tests
---
## train_svm_utils.py

Hilffunktionen fürs SVM 

## train_svm.py

Anpassen einer SVM auf dem Corpus und anzeigen der Metriken

