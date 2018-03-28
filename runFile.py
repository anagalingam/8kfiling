import textPredict
import read8K
import joblib
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import NMF
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier

trainEndDate = np.datetime64('2011-01-01')
prices = textPredict.getPriceHistory( trainEndDate )
trainDocs = joblib.load('data8K/tune')

for daysForward in range(0,5):
    textPredict.setTargets( trainDocs , prices , daysForward )

trainDocs = textPredict.getDocsWithTarget( trainDocs )
textPredict.setRecentMovements( trainDocs , prices )
textPredict.setEarningsSurprise( trainDocs , prices )
textPredict.setVolatility( trainDocs )
joblib.dump(trainDocs, 'data8K/tuneDocs')
print "Dumped trainDocs into data8K"
import sys
sys.exit()

trainDocs = joblib.load('data8K/trainDocs0')
nonTextFeatures = textPredict.getNonTextFeatures(trainDocs)
print "Got non text features"
targets = [doc['target0'] for doc in trainDocs]
# Build term doc matrix for training docs
model, termDoc = read8K.buildTermDocMatrix(trainDocs)
print "Built Term-Doc Matrix"
joblib.dump(model, 'data8K/vectModel')
print "Dumped vectorizer model into data8K/vectModel"
joblib.dump(termDoc, 'data8K/vect')
print "Dumped termDoc into data8K/vect"

tfidfTrans = TfidfTransformer()
tfidfTermDoc = tfidfTrans.fit_transform(termDoc)
joblib.dump(tfidfTrans, 'data8K/tfidfTransformer')
print "Dumped tfidfTransformer into data8K"

nmf100 = NMF(n_components=100)
nmfFeatures = nmf100.fit_transform(tfidfTermDoc)
joblib.dump(nmf100, 'data8K/nmfModel')
print "Dumped nmf100 model into data8K"

joblib.dump(nmfFeatures, 'data8K/nmf')
print "Dumped nmf features into data8K/nmf"

pmi2319 = SelectKBest(mutual_info_classif, k=2319)
pmiFeatures = pmi2319.fit_transform(termDoc, targets)
joblib.dump(pmi2319, 'data8K/pmiModel')
print "Dumped pmi2319 model into data8K"

joblib.dump(pmiFeatures, 'data8K/pmi')
print "Dumped PMI features into data8K/pmi"

features = sp.sparse.hstack([pmiFeatures, nmfFeatures, nonTextFeatures[0]])

clf = RandomForestClassifier(n_estimators=100)
clf.fit(features, targets)
joblib.dump(clf, 'data8K/randomForestClassifier')
print "Finished training classifier"
