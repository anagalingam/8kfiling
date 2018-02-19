import numpy as np
import pandas as pd
import re
import os
from bs4 import BeautifulSoup

def digitDate_to_datetime64( date ):
    assert type(date) == str
    assert len(date) == 8
    return np.datetime64(date[0:4] + "-" + date[4:6] + "-" + date[6:8])

snpNum = "500"

snpNamesFile = open("snp_list/snp"+snpNum+"_20120928.txt","r")
companies = snpNamesFile.readlines()
snpNamesFile.close()

companyInfo = []
for comp in companies:
    companyInfo.append(re.split(r'\t+',comp.rstrip('[\t\r\n]')))

companyData = pd.DataFrame(data=companyInfo, columns=["name","ticker","sector"]).set_index("ticker")

companyPrices = {}
for comp in companyData.index.values:
    companyPrices[comp] = pd.read_csv('price_history/'+comp+'.csv')
    companyPrices[comp]['Date'] = companyPrices[comp]['Date'].astype(np.datetime64)
    companyPrices[comp] = companyPrices[comp].set_index('Date')

epsFileList = os.listdir("EPS/")
totalFiles = len(epsFileList)
count = 0
percentComplete = 0
print "Loading Earnings per Share Info"
print "Progress: 0%"
for epsFile in epsFileList:
    epsDate = digitDate_to_datetime64(epsFile.split('.')[0])
    htmlDoc = open("EPS/"+epsFile,"r")
    htmlLines = htmlDoc.readlines()
    htmlDoc.close()
    htmlObj = BeautifulSoup(''.join(htmlLines), 'html.parser')
    smallTags = htmlObj.find_all('small')
    if len(smallTags) is not 0:
        del smallTags[0]    # Remove the Heading
    epsText = []
    for tag in smallTags:
        epsText.append(tag.text.encode('ascii','ignore'))
    EPS = np.reshape(filter(None,epsText),[-1,7])
    for comp in EPS:
        if comp[1] in companyPrices:
            companyPrices[comp[1]].loc[epsDate,'surprise'] = comp[2]
            companyPrices[comp[1]].loc[epsDate,'rEPS'] = comp[3]
            companyPrices[comp[1]].loc[epsDate,'cEPS'] = comp[4]
    count = count + 1
    if percentComplete < count/(totalFiles*1.0)*100:
        percentComplete += 5
        print "Progress: " + str(percentComplete) + "%"
print "Done!"

# Line to find the price metrics for adjacent days to EPS reports
# result = companyPrices['AAPL'].loc[companyPrices['AAPL'].shift(x).cEPS.notnull()]
# x positive means in the past e.g. x = 1 means previous business day

# TODO: Parse 8-K filings online
