import textPredict
import read8K
import scipy as sp
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
import os

def classifier( daysForward=0 , mode='test' ):
    if mode=='test':
        docList = joblib.load('data8K/trainDocs')
        eventTypes = joblib.load('data8K/eventTypes')
        nonTextFeatures, _ = textPredict.getNonTextFeatures(docList, eventTypes)
        nonTextFeatures = sp.sparse.csr.csr_matrix(nonTextFeatures)
        nonTextFeatures.data = np.nan_to_num(nonTextFeatures.data)
        vectModel = joblib.load('data8K/vectModel')
        tfidfTrans = joblib.load('data8K/tfidfTransformer')
        nmfModel = joblib.load('data8K/nmfModel')
        pmiModel = joblib.load('data8K/pmiModel'+str(daysForward))
        rfClf = joblib.load('data8K/randomForestClassifier'+str(daysForward))
        targets = [doc['target'+str(daysForward)] for doc in docList]
        if os.path.exists('trainTermDoc'):
            termDoc = joblib.load('trainTermDoc')
        else:
            termDoc = vectModel.transform([doc['text'] for doc in docList])
            joblib.dump(termDoc, 'trainTermDoc')
        tfidfTermDoc = tfidfTrans.transform(termDoc)
        nmfFeatures = sp.sparse.csr.csr_matrix(nmfModel.transform(tfidfTermDoc))
        pmiFeatures = pmiModel.transform(termDoc)
        features = sp.sparse.hstack([pmiFeatures, nmfFeatures, nonTextFeatures]).todense().astype('float32')
        features = np.nan_to_num(features)
        joblib.dump(features,'data8K/features'+str(daysForward))
        #print str(np.any(np.isinf(features)))
        predictions = rfClf.predict(features)
        for docNum, doc in enumerate(docList):
            doc['predict'+str(daysForward)] = predictions[docNum]
        joblib.dump(docList,'data8K/trainDocs')
        print "The train data score for "+str(daysForward) +" is: " + str(rfClf.score(features, targets))
    return

'''
trainDocs = joblib.load('data8K/trainDocs0')
nonTextFeatures = textPredict.getNonTextFeatures(trainDocs)
nmfFeatures = joblib.load('data8K/nmf0')
miFeatures = joblib.load('data8K/pmi0')

features = sp.sparse.hstack([miFeatures, nmfFeatures, nonTextFeatures])
targets = [doc['target0'] for doc in trainDocs]

clf = RandomForestClassifier(n_estimators=100)
clf.fit(features, targets)
print clf.score(features, targets)
'''
