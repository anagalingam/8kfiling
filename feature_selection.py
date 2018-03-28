import joblib
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import mutual_info_classif, SelectKBest


def getFeatures( daysForward , nmfLen , pmiLen ):
    trainDocs = joblib.load('data8K/trainDocs'+str(daysForward))
    termDoc = joblib.load('data8K/termDoc'+str(daysForward))

    targets = [doc['target'+str(daysForward)] for doc in trainDocs]

    tfModel = TfidfTransformer()
    nmfModel = NMF(int(nmfLen))

    nmfFeatures = nmfModel.fit_transform(tfModel.fit_transform(termDoc))

    pmiFeatures = SelectKBest(mutual_info_classif, k=2319).fit_transform(termDoc, targets)

    return nmfFeatures, pmiFeatures
