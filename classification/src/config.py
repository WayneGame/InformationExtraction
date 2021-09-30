import re


RE_COND_COMMENTS = r'(<!)?\[.{3,20}\]>?' # [if IE 9]><![endif]
RE_TAGS = r'<.*>(<\/...?)?' # <img id="logo" alt="Logo: Hessen - zur Startse...
RE_LINK = r'(http|ftp|https):\/\/([\w\-_]+(?:(?:\.[\w\-_]+)+))([\w\-\.,@?^=%&:/~\+#]*[\w\-\@?^=%&/~\+#])?'

SPECIAL_CHARACTERS = "!@#$%^&*()[]{};:,.<>/?\|`~-=_+»«\"\'" 
SPECIAL_CHARACTERS_BERT = "@#$%^&*()[]{};,/›‹<>\|`~-=_+»«\"\'" # TODO Satzzeichen . : ? ! entfernte für einen Bert Test

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

COVID_WORDS = ['corona', 'covid']
RESTRICTION_WORDS = ['einschränkung', 'beschränkung', 'regel', 'verordnungen', 'regelung', 'maßnahme', 'lockerung']

# SEQUENCE LENGTHS
MAX_LINK_SEQUENCE_LENGTH  = 50
MAX_TITLE_SEQUENCE_LENGTH = 50
MAX_META_SEQUENCE_LENGTH  = 50
MAX_BODY_SEQUENCE_LENGTH  = 550


STOPWORDS_PATH = "../input/stopwords_ger.csv"

MAX_NUM_WORDS = 50000
MAX_SEQUENCE_LENGTH = 2000
EMBEDDING_DIM = 700

THRESHOLD = 0.5

DATASET_PATH = "../input/web_pages_folds_2.csv"

MODEL_PATH = "../models/covid_model"
TOKENIZER_SAVE_PATH = "../models/covid_tokenizer"

USE_VACCINE = False