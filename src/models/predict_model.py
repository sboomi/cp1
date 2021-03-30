import unicodedata
import string
from nltk import stem
from nltk.corpus import stopwords


def clean_text(txt: str,
               stemmer: stem.api.StemmerI
               ) -> str:
    """Preprocesses the text to make it suitable for NLP
    algorithms. Adapted for French comments.

    Args:
        txt (str): original text
        stemmer (stem.api.StemmerI): NLTK stemmer for stemming

    Returns:
        str: cleaned text
    """
    sw = set(stopwords.words('french'))
    new_txt = ''.join([c.lower() if c not in string.punctuation else ' ' for c
                       in txt])
    new_txt = (unicodedata.normalize('NFKD', new_txt)
               .encode('ascii', 'ignore')
               .decode('utf-8', 'ignore'))
    new_txt = " ".join([word for word in new_txt.split()
                        if word and word not in sw])
    new_txt = ' '.join([stemmer.stem(word) for word in new_txt.split()])
    return new_txt


def make_ml_prediction(y_input, model):
    stemmer = stem.regexp.RegexpStemmer('s$|es$|era$|erez$|ions$| <etc> ')
    y_clean = clean_text(y_input, stemmer)
    y_pred = model.predict(y_clean)
    return y_pred
