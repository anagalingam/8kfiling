import numpy as np
import gzip
import re
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter

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
    with gzip.open('8K-gz/'+ticker+'.gz','rb') as file8k:
        fileText = file8k.read()
    docs = fileText.split('</DOCUMENT>')
    result = []
    for item in docs:
        lines = filter(None, item.split('\n'))
        if len(lines) == 0:
            continue
        thisDoc = {}
        thisDoc['Date'] = getTimestamp_from8K(lines[2])
        thisDoc['Type'] = getType_from8K(lines[3])
        thisDoc['Text'] = '\n'.join(lines[5+len(thisDoc['Type']):])
        result.append(thisDoc)
    return result

def getPOS( word ):
    if word.startswith('V'):
        return wordnet.VERB
    elif word.startswith('J'):
        return wordnet.ADJ
    elif word.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def buildUnigramModel( tickerList ):
    assert( type(tickerList) == list )
    if( len(tickerList) == 0 ):
        return None
    assert( type(tickerList[0]) == str)

    model = Counter()
    wordRegex = r'\b\w+\b'
    
    # Get word counts across all 8K docs
    for ticker in tickerList:
        companyDocs = read8KDoc( ticker )
        for doc in companyDocs:
            words = re.findall(r'\b\w+\b', doc['Text'].lower())
            for word in words:
                model[word] += 1
    
    # Remove low freq words, stop words, and numbers
    stopWords = stopwords.words('english')

    for word in model.keys():
        if model[word] < 10:
            del model[word]
        if word in stopWords:
            del model[word]
        if not re.search(r'[a-zA-Z]',word):
            del model[word]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    for word in model.keys():
        newWord = lemmatizer.lemmatize(word, getPOS(word))
        if( word != newWord ):
            count = model[word]
            del model[word]
            if newWord in model.keys():
                model[newWord] += count
            else:
                model[newWord] = count
    return model


