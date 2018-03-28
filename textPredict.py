import numpy as np
import pandas as pd
import re
import os
from bs4 import BeautifulSoup

def digitDate_to_datetime64( date ):
    assert isinstance(date,str)
    assert len(date) == 8
    return np.datetime64(date[0:4] + "-" + date[4:6] + "-" + date[6:8])

def getAllCompanyInfo( snpNum = "500" ):
    assert isinstance(snpNum,str)
    assert (snpNum == "500" or snpNum == "1000" or snpNum =="1500")
    
    # Get the name, ticker, sector for all companies in the index as a dataframe
    with open("snp_list/snp"+snpNum+"_20120928.txt","r") as snpNamesFile:
        companies = snpNamesFile.readlines()
    companyInfo = []
    for comp in companies:
        companyInfo.append(re.split(r'\t+',comp.rstrip('[\t\r\n]')))    
    return pd.DataFrame(data=companyInfo, columns=["name","ticker","sector"]).set_index("ticker")

def getCompanyPrices( ticker ):
    prices = pd.read_csv('price_history/'+ticker+'.csv')
    prices.columns = map(str.lower, prices.columns)
    prices['date'] = prices['date'].astype(np.datetime64)
    prices = prices.set_index('date')
    return prices

def getAllCompanyPrices( tickerList ):
    companyPrices = {}
    for ticker in tickerList:
        companyPrices[ticker] = getCompanyPrices(ticker)
    return companyPrices

# Used for train/test splitting data
def filterKnownPriceHistory( companyPrices , date ):
    assert isinstance(date,np.datetime64)
    
    knownPrices = {}
    
    for comp in companyPrices.keys():
        knownPrices[comp] = companyPrices[comp].loc[companyPrices[comp].index.values < date]
    return knownPrices

# Some docs are released on non-trading days, so this function finds the next day of impact
def findNextTradingDate_fromDoc( date , prices ):
    assert isinstance( date , np.datetime64 )
    assert isinstance( prices , pd.DataFrame )
    
    dateCopy = date
    
    if len(prices.index.values) == 0:
        return None
    if( dateCopy < prices.index.values[-1] ):
        return None
    while( dateCopy not in prices.index.values ):
        dateCopy = dateCopy + np.timedelta64(1,'D')
        if dateCopy > prices.index.values[0]:
            return None
    return dateCopy

def setTargets( docList , companyPrices , daysForward , indexTicker='gspc'):
    # NOTE: changes made directly to docList
    indexPrices = getCompanyPrices(indexTicker)

    for doc in docList:
        date = findNextTradingDate_fromDoc(doc['date'],companyPrices[doc['ticker']])
        if date == None:
            continue
        docLoc = companyPrices[doc['ticker']].index.get_loc(date)
        # skip this doc if we don't have the required price history to assign a target
        if docLoc + 1 == len(companyPrices[doc['ticker']].index.values):
            continue
        if docLoc - daysForward < 0:
            continue
        
        indexLoc = indexPrices.index.get_loc(date)
        indexForwardPrice = indexPrices.iloc[indexLoc-daysForward].loc['adj close']
        indexInitPrice = indexPrices.iloc[indexLoc+1].loc['adj close']
        indexChangePercent = (indexForwardPrice - indexInitPrice)/indexInitPrice*100
        
        docForwardPrice = companyPrices[doc['ticker']].iloc[docLoc-daysForward].loc['adj close']
        docInitPrice = companyPrices[doc['ticker']].iloc[docLoc+1].loc['adj close']
        docChangePercent = (docForwardPrice - docInitPrice)/docInitPrice*100
        
        adjDocImpact = docChangePercent-indexChangePercent
        
        if adjDocImpact > 1.0:
            doc['target'+str(daysForward)] = 1
        elif adjDocImpact < -1.0:
            doc['target'+str(daysForward)] = -1
        else:
            doc['target'+str(daysForward)] = 0
    return

def getDocsWithTarget( docList ):
    res = []
    targets = ["target"+str(daysForward) for daysForward in range(0,5)]
    for doc in docList:
        for target in targets:
            if target in doc.keys():
                res.append(doc)
                break
    return res

def setEPS( companyPrices ):
    # NOTE: changes made directly to companyPrices
    epsFileList = os.listdir("EPS/")

    # Since this takes a while, include print statements on progress
    totalFiles = len(epsFileList)
    count = 0
    percentComplete = 0
    print "Loading Earnings per Share Info"
    print "Progress: 0%"

    for epsFile in epsFileList:
        epsDate = digitDate_to_datetime64(epsFile.split('.')[0])
        with open("EPS/"+epsFile,"r") as htmlDoc:
            htmlLines = htmlDoc.readlines()
        htmlObj = BeautifulSoup(''.join(htmlLines), 'html.parser')
        smallTags = htmlObj.find_all('small')   # Desired data in <small> tags
        if len(smallTags) is not 0:
            del smallTags[0]    # Remove the Heading

        # Get EPS table data and store in corresponding companyPrices
        epsText = []
        for tag in smallTags:
            epsText.append(tag.text.encode('ascii','ignore'))
        EPS = np.reshape(filter(None,epsText),[-1,7])
        for comp in EPS:
            if comp[1] in companyPrices:
                date = findNextTradingDate_fromDoc(epsDate,companyPrices[comp[1]])
                if date is None:
                    continue
                companyPrices[comp[1]].loc[date,'surprise'] = comp[2]
                companyPrices[comp[1]].loc[date,'rEPS'] = comp[3]
                companyPrices[comp[1]].loc[date,'cEPS'] = comp[4]
        
        # Update progress
        count = count + 1
        if percentComplete < count/(totalFiles*1.0)*100:
            percentComplete += 5
            print "Progress: " + str(percentComplete) + "%"
    return

def getPriceHistory( date , snpNum = "500" , addEPS = True):
    info = getAllCompanyInfo(snpNum)
    prices = getAllCompanyPrices(info.index.values)
    if addEPS:
        setEPS(prices)
    knownPrices = filterKnownPriceHistory(prices, date)
    return prices

def setRecentMovements( docList , companyPrices , indexTicker='gspc'):
    # NOTE: changes made directly to docList
    indexPrices = getCompanyPrices(indexTicker)
    for doc in docList:
        date = findNextTradingDate_fromDoc(doc['date'],companyPrices[doc['ticker']])
        if date == None:
            continue
        docLoc = companyPrices[doc['ticker']].index.get_loc(date)
        indexLoc = indexPrices.index.get_loc(date)
        adjPricesBack = []
        for daysBack in [5,20,65,250]:
            if docLoc + daysBack >= len(companyPrices[doc['ticker']]):
                compPriceBack = companyPrices[doc['ticker']].iloc[-1].loc['adj close']
            else:
                compPriceBack = companyPrices[doc['ticker']].iloc[docLoc+daysBack].loc['adj close']
            compPriceNow = companyPrices[doc['ticker']].iloc[docLoc].loc['adj close']
            compChangePercent = (compPriceNow - compPriceBack)/compPriceBack*100

            if indexLoc + daysBack >= len(indexPrices):
                indexPriceBack = indexPrices.iloc[-1].loc['adj close']
            else:
                indexPriceBack = indexPrices.iloc[indexLoc+daysBack].loc['adj close']
            indexPriceNow = indexPrices.iloc[indexLoc+1].loc['adj close']
            indexChangePercent = (indexPriceNow - indexPriceBack)/indexPriceBack*100
            
            adjPricesBack.append(compChangePercent - indexChangePercent)

        doc['recentW'] = adjPricesBack[0]
        doc['recentM'] = adjPricesBack[1]
        doc['recentQ'] = adjPricesBack[2]
        doc['recentY'] = adjPricesBack[3]
    return

def setEarningsSurprise( docList , companyPrices ):
    for doc in docList:
        date = findNextTradingDate_fromDoc(doc['date'], companyPrices[doc['ticker']])
        if date is None:
            continue

        # if this company does NOT have a valid surprise value for this date, set doc surprise to 0
        if('surprise' not in companyPrices[doc['ticker']].loc[date]):
            doc['surprise'] = 0
        elif(pd.isnull(companyPrices[doc['ticker']].loc[date,'surprise'])):
            doc['surprise'] = 0
        elif(isinstance(companyPrices[doc['ticker']].loc[date,'surprise'],str)):
            doc['surprise'] = 0
        else:
            doc['surprise'] = companyPrices[doc['ticker']].loc[date,'surprise']
    return

def filterDocsWithEPS( docList ):
    filtered = []
    for doc in docList:
        if not pd.isnull(doc['surprise']):
            filtered.append(doc)
    return filtered

def setVolatility( docList , indexTicker='vix' ):
    indexPrices = getCompanyPrices(indexTicker)
    for doc in docList:
        date = findNextTradingDate_fromDoc(doc['date'], indexPrices)
        if date is None:
            continue
        doc['volatility'] = indexPrices.loc[date,'adj close']
    return

def getNonTextFeatures( docList , types=None ):
    eventTypes = {}
    if types == None:
        for doc in docList:
            for docType in doc['type']:
                if docType not in eventTypes.keys():
                    eventTypes[docType] = 1
                else:
                    eventTypes[docType] += 1
        for docType in eventTypes.keys():
            if eventTypes[docType] < 10:
                del eventTypes[docType]
    else:
        for docType in types:
            eventTypes[docType] = 0
        for doc in docList:
            for docType in doc['type']:
                if docType in eventTypes.keys():
                    eventTypes[docType] += 1

    features = np.zeros((len(docList),6+len(eventTypes.keys())))

    for docNum in range(len(docList)):
        features[docNum,0] = docList[docNum]['surprise']
        features[docNum,1] = docList[docNum]['volatility']
        features[docNum,2] = docList[docNum]['recentW']
        features[docNum,3] = docList[docNum]['recentM']
        features[docNum,4] = docList[docNum]['recentQ']
        features[docNum,5] = docList[docNum]['recentY']
        eventNum = 0
        for docType in eventTypes.keys():
            if docType in docList[docNum]:
                features[docNum,6+eventNum] = 1
            eventNum += 1
    return features, eventTypes.keys() 

def featurizeDataset( dataset , date , daysForward ):
    prices = getPriceHistory( date )
    docList = joblib.load('data8K/'+dataset)
    setTargets( docList , prices , daysForward )
    docList = getDocsWithTarget( docList , daysForward )
    setRecentMovements( docList , prices )
    setEarningsSurprise( docList , prices )
    setVolatility( docList )
    joblib.dump( docList , 'data8K/'+dataset+str(daysForward) )
    return

def getCompanyTimeFeatures( filteredPrices , docList ):
    desiredCol = ['adj close','volume','predict','spread']
    
    for comp in filteredPrices:
        if 'predict' not in filteredPrices[comp]:
            filteredPrices[comp]['predict'] = 0
        if 'spread' not in filteredPrices[comp]:
            filteredPrices[comp]['spread'] = (filteredPrices[comp]['high']/filteredPrices[comp]['low'])-1
        for col in filteredPrices[comp].columns:
            if col not in desiredCol:
                filteredPrices[comp].drop(col,1,inplace=True)
    
    for doc in docList:
        date = findNextTradingDate_fromDoc(doc['date'], filteredPrices[doc['ticker']])
        dateIndex = filteredPrices[doc['ticker']].index.get_loc(date)+1
        filteredPrices[doc['ticker']].iloc[dateIndex] = doc['predict0']

    return filteredPrices


# Line to find the price metrics for adjacent days to EPS reports
# result = companyPrices['AAPL'].loc[companyPrices['AAPL'].shift(x).cEPS.notnull()]
# x positive means in the past e.g. x = 1 means previous business day
