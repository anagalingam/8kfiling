import numpy as np
import gzip
import re
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer


def getPOS( posTag ):
    if posTag.startswith('N'):
        return wordnet.NOUN
    elif posTag.startswith('V'):
        return wordnet.VERB
    elif posTag.startswith('J'):
        return wordnet.ADJ
    elif posTag.startswith('R'):
        return wordnet.ADV
    else:
        return None

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self,text):
        return [self.wnl.lemmatize(w[0], getPOS(w[1])) for w in filter(lambda x: getPOS(x[1]),pos_tag(word_tokenize(text)))]

def getTimestamp_from8K( timeString ):
    assert( type(timeString) == str )
    assert( len(timeString) == 19 )
    return np.datetime64(timeString[5:9] + '-' + timeString[9:11] + '-' + timeString[11:13])

def getType_from8K( eventString ):
    assert( type(eventString) == str )
    assert( eventString[0:6] == 'EVENTS' )
    res = eventString.split('\t')
    return res[1:]

def read8KDoc( ticker ):
    assert isinstance(ticker,str)

    with gzip.open('8K-gz/'+ticker+'.gz','rb') as file8k:
        fileText = file8k.read()
    docs = fileText.split('</DOCUMENT>')
    result = []
    for item in docs:
        lines = filter(None, item.split('\n'))
        if len(lines) == 0:
            continue
        thisDoc = {}
        thisDoc['ticker'] = ticker
        thisDoc['date'] = getTimestamp_from8K(lines[2])
        thisDoc['type'] = getType_from8K(lines[3])
        thisDoc['text'] = '\n'.join(lines[5+len(thisDoc['type']):])
        result.append(thisDoc)
    return result

def getAll8KDoc( tickerList ):
    res = []
    for ticker in tickerList:
        companyDocs = read8KDoc( ticker )
        for doc in companyDocs:
            res.append(doc)
    return res

# Splits on doc timestamps < date for train/test split
def split8KDocs( docList , date ):
    assert isinstance(docList, list)
    assert isinstance(date, np.datetime64)

    before = []
    after = []

    for doc in docList:
        if doc['date'] < date:
            before.append(doc)
        else:
            after.append(doc)
    
    return before, after

def buildTermDocMatrix( docList ):
    vect = CountVectorizer(tokenizer=LemmaTokenizer())
    termDoc = vect.fit_transform([doc['text'] for doc in docList])
    return vect, termDoc

