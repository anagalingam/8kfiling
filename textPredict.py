import numpy as np
import pandas as pd
import re
import os
from bs4 import BeautifulSoup

def digitDate_to_datetime64( date ):
    assert isinstance(date,str)
    assert len(date) == 8
    return np.datetime64(date[0:4] + "-" + date[4:6] + "-" + date[6:8])

def getCompanyInfo( snpNum = "500" ):
    assert isinstance(snpNum,str)
    assert (snpNum == "500" or snpNum == "1000" or snpNum =="1500")
    
    # Get the name, ticker, sector for all companies in the index as a dataframe
    with open("snp_list/snp"+snpNum+"_20120928.txt","r") as snpNamesFile:
        companies = snpNamesFile.readlines()
    companyInfo = []
    for comp in companies:
        companyInfo.append(re.split(r'\t+',comp.rstrip('[\t\r\n]')))    
    return pd.DataFrame(data=companyInfo, columns=["name","ticker","sector"]).set_index("ticker")
    
def getPriceHistory( companyInfo ):
    assert isinstance(companyInfo,pd.DataFrame)
    # Get the price histories for each ticker as a dictionary
    companyPrices = {}
    for comp in companyInfo.index.values:
        companyPrices[comp] = pd.read_csv('price_history/'+comp+'.csv')
        companyPrices[comp].columns = map(str.lower, companyPrices[comp].columns)
        companyPrices[comp]['date'] = companyPrices[comp]['date'].astype(np.datetime64)
        companyPrices[comp] = companyPrices[comp].set_index('date')
    return companyPrices

# Used for splitting data
def getKnownPriceHistory( companyPrices , date ):
    assert isinstance(date,np.datetime64)
    knownPrices = {}
    for comp in companyPrices.keys():
        knownPrices[comp] = companyPrices[comp].loc[companyPrices[comp].index.values < date]
    return knownPrices

def readEPS( companyPrices ):
    # NOTE: changes made directly to input argument
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
                companyPrices[comp[1]].loc[epsDate,'surprise'] = comp[2]
                companyPrices[comp[1]].loc[epsDate,'rEPS'] = comp[3]
                companyPrices[comp[1]].loc[epsDate,'cEPS'] = comp[4]
        
        # Update progress
        count = count + 1
        if percentComplete < count/(totalFiles*1.0)*100:
            percentComplete += 5
            print "Progress: " + str(percentComplete) + "%"
    return

# Line to find the price metrics for adjacent days to EPS reports
# result = companyPrices['AAPL'].loc[companyPrices['AAPL'].shift(x).cEPS.notnull()]
# x positive means in the past e.g. x = 1 means previous business day

