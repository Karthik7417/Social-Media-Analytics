import string
import warnings
warnings.filterwarnings('ignore')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def cleaning(data):
    #Remove whitespaces
    # remove whitespaces and links
    data["text"] = data["text"].str.replace('http\S+|www.\S+', '', case=False)
    data['text'] = data['text'].str.strip()
    #
    # lowercase the text
    data['text'] = data['text'].str.lower()

    # remove punctuation
    punc = string.punctuation
    table = str.maketrans('', '', punc)
    data['text'] = data['text'].apply(lambda x: x.translate(table))

    # tokenizing each message
    data['word_tokens'] = data.apply(lambda x: x['text'].split(' '), axis=1)

    # removing stopwords
    cachedStopWords = stopwords.words("english")
    cachedStopWords.append('opioid')
    cachedStopWords.append('crisis')
    cachedStopWords.append('drug')
    cachedStopWords.append('usage')

    data['cleaned_text'] = data.apply(lambda x: [word for word in x['word_tokens'] if word not in cachedStopWords],axis=1)

    data['word_tokens'] = [list(zip(x, x[1:])) for x in data.cleaned_text.values.tolist()]
    data['word_tokens'] = data.apply(lambda x: (list(' '.join(w) for w in x['word_tokens'])), axis=1)


    # stemming
    #ps = PorterStemmer()
    #data['stemmed'] = data.apply(lambda x: [ps.stem(word) for word in x['cleaned_text']], axis=1)

    # remove single letter words
    data['final_text'] = data.apply(lambda x: ' '.join([word for word in x['cleaned_text'] if len(word) > 1]), axis=1)

    # label encoding negative tweets=0 and gun violence tweets=1
    data.loc[data['type'] == 'negative tweets', 'type'] = 0
    data.loc[data['type'] == 'opoid crisis', 'type'] = 1

    return data[['type', 'text', 'final_text']]