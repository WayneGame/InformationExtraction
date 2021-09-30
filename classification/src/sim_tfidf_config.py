
CORPUS_SAVE_PATH = "../input/similarity_corpus.csv"
TEST_PATH = "../input/testfragen.csv"
BERT_PATH = "../input/sim_bert_model_weights_2epochs.h5"
BERT_MODEL = "bert-base-german-cased"

# Wie soll der Corpus für TF/IDF, BM25 aufgebaut sein?
# bzw. gegen welchen Teil der Webseiten wird die Nutzerfrage verglichen?
CORPUS_COLUMNS = ["link", "body", "meta", "title", "h_elements"]

COVID_WORDS = ["corona", "cov"]
VACCI_WORDS = ["impf"]
CONST_WORDS = ["einschränkung", "anordnungen"]
SCHOOL      = ["schule"]
CHILDREN    = ["kind", "kita"]


# TODO!!!!: hier nochmal nachdenken, ob ein dict sinnvoller ist. => eher nein, wenn die Anfrage genauso behandelt wird.
# hmm ggf wäre eine Thematische Umwandlung sinnvoller Tochter => Kind, Neffe => Kind, Uni => Schule
# TODO ggf auch seperat: einmal WORD LIST und einmal Themendict { "kind": ["sohn", "tochter", ...], "schule": ["uni", ...]}
WORD_LIST = [
    COVID_WORDS,
    VACCI_WORDS,
    CONST_WORDS,
    SCHOOL,
    CHILDREN
    ]

THEMEN = { 
    "kind"  : ["sohn", "tochter", "neffe", "nichte", "bruder", "schwester", "kita", "familie", "ferien", "kindergarten"], 
    "schule": ["uni", "mensa", "semester", "studium", "prüfung", "schüler"]
    }

# TODO: Zuprnung Städte zu Ländern
# Probleme mit Plural (z.B. Museen, Schwimmbäder, ...), etc. 