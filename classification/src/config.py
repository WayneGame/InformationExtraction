import re


RE_COND_COMMENTS = r'(<!)?\[.{3,20}\]>?' # z.B. [if IE 9]><![endif]
RE_TAGS = r'<.*>(<\/...?)?' # z.B. <img id="logo" alt="Logo: Hessen - zur Startse...
RE_LINK = r'(http|ftp|https):\/\/([\w\-_]+(?:(?:\.[\w\-_]+)+))([\w\-\.,@?^=%&:/~\+#]*[\w\-\@?^=%&/~\+#])?'

SPECIAL_CHARACTERS = "!@#$%^&*()[]{};:,.<>/?\|`~-=_+»«\"\'" 
SPECIAL_CHARACTERS_BERT = "@#$%^&*()[]{};,/›‹<>\|`~-=_+»«\"\'" # Für Sentence-Bert brauchen wir die Satzzeichen . : ? ! 

TAG_BLACKLIST = [
        '[document]',
        'noscript',
        'header',   
        'html',
        'meta',
        'head', 
        'input',
        'script',
        'style',
        'title',
        'nav',
        'footer',
        'img',
        'svg'
    ]

EPOCHS = 50
K_FOLD_SPLITS = 10

BATCH_SIZE = 64

"""
Die Worte für covid_word_count, etc.
"""
COVID_WORDS = ['corona', 'covid']
RESTRICTION_WORDS = ['einschränkung', 'beschränkung', 'regel', 'verordnungen', 'regelung', 'maßnahme', 'lockerung']


MAX_LINK_SEQUENCE_LENGTH  = 50
MAX_TITLE_SEQUENCE_LENGTH = 50
MAX_META_SEQUENCE_LENGTH  = 50
MAX_BODY_SEQUENCE_LENGTH  = 550


STOPWORDS_PATH = "../input/stopwords_ger.csv"

MAX_NUM_WORDS = 50000
MAX_SEQUENCE_LENGTH = 2000
EMBEDDING_DIM = 700

# Speicherort für den verarbeiteten Datensatz
DATASET_PATH = "../input/web_pages_folds_2.csv"

MODEL_PATH = "../models/covid_model"
TOKENIZER_SAVE_PATH = "../models/covid_tokenizer"

USE_VACCINE = False