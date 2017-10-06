import os
import sys
import pickle
#from nltk.corpus.reader.wordnet import Lemma
import codecs
import networkx as nx
import re
from pprint import pprint
import matplotlib.pyplot as plt
import pandas as pd
import copy
import operator
import collections
import itertools
from colorama import Fore, Back, Style
from sentences import  getSentenceDFTagList, getResultsDFTagList, TXConvert

from nltk.stem.wordnet import WordNetLemmatizer
lmtzr=WordNetLemmatizer()#create a lemmatizer object

import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def getSourceDestKindForRelation(source, dest, rel, parentConnectionExceptions={}):
    if not parentConnectionExceptions:
        parentConnectionExceptions["txBeComparedTo txNonPred narg source"] = 1.00
        parentConnectionExceptions["txBeFrom txNamed narg source"] = 1.00
        parentConnectionExceptions["txBeLocatedAt txFrequentDouble arg dest"] = 1.00
        parentConnectionExceptions["txBeLocatedAt txNamed narg source"] = 1.00
        parentConnectionExceptions["txBeLocatedAt txNonPred narg source"] = 0.72
        parentConnectionExceptions["txBeLocatedAt txPerson0 arg dest"] = 1.00
        parentConnectionExceptions["txBeLocatedAt txPerson1 arg dest"] = 1.00
        parentConnectionExceptions["txBeLocatedAt txThing1 arg dest"] = 1.00
        parentConnectionExceptions["txBeTemporallyAt txNonPred narg source"] = 0.60
        parentConnectionExceptions["txBeTemporallyAt txPerson0 arg dest"] = 1.00
        parentConnectionExceptions["txBeTemporallyAt txPred narg source"] = 1.00
        parentConnectionExceptions["txFrequentDouble txNamed narg source"] = 0.86
        parentConnectionExceptions["txFrequentDouble txNonPred narg source"] = 0.98
        parentConnectionExceptions["txHaveConcession txDateEntity narg source"] = 1.00
        parentConnectionExceptions["txHaveConcession txFrequentDouble arg dest"] = 1.00
        parentConnectionExceptions["txHaveConcession txFrequentDouble arg source"] = 1.00
        parentConnectionExceptions["txHaveConcession txHaveCondition arg source"] = 1.00
        parentConnectionExceptions["txHaveConcession txNamed arg source"] = 1.00
        parentConnectionExceptions["txHaveConcession txNamed narg source"] = 1.00
        parentConnectionExceptions["txHaveConcession txNonPred arg source"] = 0.90
        parentConnectionExceptions["txHaveConcession txNonPred narg source"] = 0.93
        parentConnectionExceptions["txHaveConcession txPerson0 narg source"] = 1.00
        parentConnectionExceptions["txHaveConcession txPerson1 arg source"] = 0.67
        parentConnectionExceptions["txHaveConcession txPerson1 narg source"] = 1.00
        parentConnectionExceptions["txHaveConcession txPred narg source"] = 1.00
        parentConnectionExceptions["txHaveConcession txThing1 arg dest"] = 0.75
        parentConnectionExceptions["txHaveCondition txNamed arg source"] = 1.00
        parentConnectionExceptions["txHaveCondition txNamed narg source"] = 1.00
        parentConnectionExceptions["txHaveCondition txNonPred arg source"] = 0.88
        parentConnectionExceptions["txHaveCondition txPerson0 arg dest"] = 1.00
        parentConnectionExceptions["txHaveCondition txPerson0 arg source"] = 1.00
        parentConnectionExceptions["txHaveCondition txPerson2 arg dest"] = 1.00
        parentConnectionExceptions["txHaveCondition txPerson2 arg source"] = 1.00
        parentConnectionExceptions["txHaveCondition txPred narg source"] = 1.00
        parentConnectionExceptions["txHaveCondition txThing1 arg dest"] = 1.00
        parentConnectionExceptions["txHaveCondition txThing1 arg source"] = 1.00
        parentConnectionExceptions["txHaveManner txNonPred arg source"] = 1.00
        parentConnectionExceptions["txHaveManner txNonPred narg source"] = 1.00
        parentConnectionExceptions["txHaveOrg txFrequentDouble arg dest"] = 1.00
        parentConnectionExceptions["txHaveOrg txHaveRel arg dest"] = 1.00
        parentConnectionExceptions["txHaveOrg txNamed narg source"] = 0.94
        parentConnectionExceptions["txHaveOrg txNonPred narg source"] = 0.89
        parentConnectionExceptions["txHaveOrg txPerson0 arg dest"] = 1.00
        parentConnectionExceptions["txHaveOrg txPerson1 arg dest"] = 1.00
        parentConnectionExceptions["txHaveOrg txPerson2 arg dest"] = 1.00
        parentConnectionExceptions["txHaveOrg txPred narg source"] = 0.93
        parentConnectionExceptions["txHaveOrg txTemporalQuantity narg source"] = 1.00
        parentConnectionExceptions["txHavePart txNamed narg source"] = 1.00
        parentConnectionExceptions["txHavePart txNonPred narg source"] = 0.83
        parentConnectionExceptions["txHavePart txPred narg source"] = 1.00
        parentConnectionExceptions["txHavePurpose txHaveConcession arg dest"] = 1.00
        parentConnectionExceptions["txHavePurpose txHaveConcession arg source"] = 1.00
        parentConnectionExceptions["txHavePurpose txNamed arg source"] = 0.62
        parentConnectionExceptions["txHavePurpose txNonPred arg source"] = 0.61
        parentConnectionExceptions["txHavePurpose txNonPred narg source"] = 0.55
        parentConnectionExceptions["txHavePurpose txPred narg source"] = 1.00
        parentConnectionExceptions["txHavePurpose txThing1 arg dest"] = 1.00
        parentConnectionExceptions["txHavePurpose txThing1 arg source"] = 1.00
        parentConnectionExceptions["txHaveQuant txNonPred narg source"] = 1.00
        parentConnectionExceptions["txHaveRel txMost narg source"] = 1.00
        parentConnectionExceptions["txHaveRel txNonPred narg source"] = 0.55
        parentConnectionExceptions["txHaveRel txPerson0 arg dest"] = 1.00
        parentConnectionExceptions["txHaveRel txPerson1 arg dest"] = 1.00
        parentConnectionExceptions["txHaveRel txPerson2 arg dest"] = 1.00
        parentConnectionExceptions["txHaveSubevent txFrequentDouble arg dest"] = 1.00
        parentConnectionExceptions["txHaveSubevent txFrequentDouble arg source"] = 1.00
        parentConnectionExceptions["txHaveSubevent txNamed arg source"] = 1.00
        parentConnectionExceptions["txHaveSubevent txNonPred arg source"] = 1.00
        parentConnectionExceptions["txHaveSubevent txNonPred narg source"] = 1.00
        parentConnectionExceptions["txInclude txFrequentDouble arg dest"] = 1.00
        parentConnectionExceptions["txInclude txFrequentDouble arg source"] = 1.00
        parentConnectionExceptions["txInclude txFrequentDouble narg dest"] = 1.00
        parentConnectionExceptions["txInclude txFrequentDouble narg source"] = 1.00
        parentConnectionExceptions["txInclude txInclude arg dest"] = 1.00
        parentConnectionExceptions["txInclude txMonetaryQuantity narg source"] = 1.00
        parentConnectionExceptions["txInclude txMore narg source"] = 1.00
        parentConnectionExceptions["txInclude txMost narg source"] = 1.00
        parentConnectionExceptions["txInclude txNamed narg source"] = 1.00
        parentConnectionExceptions["txInclude txNonPred narg source"] = 0.94
        parentConnectionExceptions["txInclude txPerson0 arg dest"] = 1.00
        parentConnectionExceptions["txInclude txPerson1 arg dest"] = 1.00
        parentConnectionExceptions["txInclude txPerson2 arg dest"] = 1.00
        parentConnectionExceptions["txInclude txPred narg source"] = 0.91
        parentConnectionExceptions["txInclude txThing0 arg dest"] = 1.00
        parentConnectionExceptions["txInclude txThing1 arg dest"] = 1.00
        parentConnectionExceptions["txInclude txThing1 narg dest"] = 1.00
        parentConnectionExceptions["txInclude txThing1 narg source"] = 1.00
        parentConnectionExceptions["txInsteadOf txFrequentDouble arg dest"] = 1.00
        parentConnectionExceptions["txInsteadOf txFrequentDouble arg source"] = 1.00
        parentConnectionExceptions["txInsteadOf txNamed arg source"] = 1.00
        parentConnectionExceptions["txInsteadOf txNamed narg source"] = 1.00
        parentConnectionExceptions["txInsteadOf txNonPred arg source"] = 0.82
        parentConnectionExceptions["txInsteadOf txNonPred narg source"] = 1.00
        parentConnectionExceptions["txInsteadOf txRateEntity arg source"] = 1.00
        parentConnectionExceptions["txMonetaryQuantity txNamed narg source"] = 1.00
        parentConnectionExceptions["txMonetaryQuantity txPerson0 narg dest"] = 1.00
        parentConnectionExceptions["txMonetaryQuantity txPerson2 narg dest"] = 1.00
        parentConnectionExceptions["txMore txBeLocatedAt arg dest"] = 1.00
        parentConnectionExceptions["txMore txMonetaryQuantity narg source"] = 1.00
        parentConnectionExceptions["txMore txPerson0 arg dest"] = 1.00
        parentConnectionExceptions["txMore txThing0 arg dest"] = 1.00
        parentConnectionExceptions["txMore txThing1 arg dest"] = 1.00
        parentConnectionExceptions["txMore txThing1 narg dest"] = 1.00
        parentConnectionExceptions["txMore txThing2 arg dest"] = 1.00
        parentConnectionExceptions["txMost txInclude arg dest"] = 1.00
        parentConnectionExceptions["txMost txPerson1 narg dest"] = 1.00
        parentConnectionExceptions["txMost txThing1 narg dest"] = 1.00
        parentConnectionExceptions["txNamed txBeLocatedAt narg dest"] = 1.00
        parentConnectionExceptions["txNamed txFrequentDouble narg dest"] = 1.00
        parentConnectionExceptions["txNamed txPerson0 narg dest"] = 1.00
        parentConnectionExceptions["txNonPred txFrequentDouble narg dest"] = 1.00
        parentConnectionExceptions["txNonPred txHaveConcession narg dest"] = 0.60
        parentConnectionExceptions["txNonPred txHaveOrg narg dest"] = 0.55
        parentConnectionExceptions["txNonPred txInclude narg dest"] = 1.00
        parentConnectionExceptions["txNonPred txPerson0 narg dest"] = 0.72
        parentConnectionExceptions["txNonPred txThing0 narg dest"] = 1.00
        parentConnectionExceptions["txNonPred txThing1 narg dest"] = 0.80
        parentConnectionExceptions["txNonPred txThing2 narg dest"] = 0.93
        parentConnectionExceptions["txPerson0 txFrequentDouble arg dest"] = 1.00
        parentConnectionExceptions["txPerson0 txHaveOrg arg dest"] = 1.00
        parentConnectionExceptions["txPerson0 txMore narg source"] = 1.00
        parentConnectionExceptions["txPerson0 txNamed narg source"] = 0.88
        parentConnectionExceptions["txPerson0 txNonPred narg source"] = 0.71
        parentConnectionExceptions["txPerson0 txPerson0 narg dest"] = 1.00
        parentConnectionExceptions["txPerson0 txPerson0 narg source"] = 1.00
        parentConnectionExceptions["txPerson0 txThing1 arg dest"] = 1.00
        parentConnectionExceptions["txPerson0 txThing2 arg dest"] = 1.00
        parentConnectionExceptions["txPerson1 txBeLocatedAt narg source"] = 1.00
        parentConnectionExceptions["txPerson1 txFrequentDouble arg dest"] = 1.00
        parentConnectionExceptions["txPerson1 txMost narg source"] = 1.00
        parentConnectionExceptions["txPerson1 txNamed narg source"] = 0.76
        parentConnectionExceptions["txPerson1 txNonPred narg source"] = 0.58
        parentConnectionExceptions["txPerson1 txPerson0 arg dest"] = 1.00
        parentConnectionExceptions["txPerson2 txHaveCondition arg dest"] = 1.00
        parentConnectionExceptions["txPerson2 txNamed narg source"] = 0.92
        parentConnectionExceptions["txPerson2 txNonPred narg source"] = 0.62
        parentConnectionExceptions["txPerson2 txPerson0 arg dest"] = 1.00
        parentConnectionExceptions["txPerson2 txPerson1 arg dest"] = 1.00
        parentConnectionExceptions["txPerson2 txThing1 arg dest"] = 1.00
        parentConnectionExceptions["txPred txBeTemporallyAt arg dest"] = 0.60
        parentConnectionExceptions["txPred txBeTemporallyAt narg dest"] = 1.00
        parentConnectionExceptions["txPred txFrequentDouble arg dest"] = 0.99
        parentConnectionExceptions["txPred txFrequentDouble narg dest"] = 1.00
        parentConnectionExceptions["txPred txHaveConcession arg dest"] = 0.77
        parentConnectionExceptions["txPred txHaveOrg arg dest"] = 0.80
        parentConnectionExceptions["txPred txHavePart arg dest"] = 0.70
        parentConnectionExceptions["txPred txHaveRel arg dest"] = 0.58
        parentConnectionExceptions["txPred txInclude arg dest"] = 0.89
        parentConnectionExceptions["txPred txInclude narg dest"] = 1.00
        parentConnectionExceptions["txPred txPerson0 arg dest"] = 0.80
        parentConnectionExceptions["txPred txPerson1 arg dest"] = 0.63
        parentConnectionExceptions["txPred txPerson2 arg dest"] = 0.69
        parentConnectionExceptions["txPred txRateEntity arg dest"] = 0.67
        parentConnectionExceptions["txPred txThing0 arg dest"] = 0.84
        parentConnectionExceptions["txPred txThing0 narg dest"] = 1.00
        parentConnectionExceptions["txPred txThing1 arg dest"] = 0.79
        parentConnectionExceptions["txPred txThing1 narg dest"] = 0.79
        parentConnectionExceptions["txPred txThing2 arg dest"] = 0.95
        parentConnectionExceptions["txPred txThing2 narg dest"] = 1.00
        parentConnectionExceptions["txPred txWhy arg dest"] = 0.56
        parentConnectionExceptions["txRateEntity txNamed narg source"] = 1.00
        parentConnectionExceptions["txThing0 txNamed narg source"] = 1.00
        parentConnectionExceptions["txThing0 txNonPred narg source"] = 0.57
        parentConnectionExceptions["txThing0 txPerson1 narg dest"] = 1.00
        parentConnectionExceptions["txThing0 txPerson1 narg source"] = 1.00
        parentConnectionExceptions["txThing0 txPred narg source"] = 0.75
        parentConnectionExceptions["txThing1 txFrequentDouble arg dest"] = 1.00
        parentConnectionExceptions["txThing1 txMore narg source"] = 1.00
        parentConnectionExceptions["txThing1 txMost narg source"] = 1.00
        parentConnectionExceptions["txThing1 txNonPred narg source"] = 0.73
        parentConnectionExceptions["txThing1 txPerson0 arg dest"] = 1.00
        parentConnectionExceptions["txThing1 txPerson0 narg source"] = 1.00
        parentConnectionExceptions["txThing1 txPerson1 arg dest"] = 1.00
        parentConnectionExceptions["txThing1 txPred narg source"] = 0.77
        parentConnectionExceptions["txThing1 txThing1 arg dest"] = 1.00
        parentConnectionExceptions["txThing2 txFrequentDouble arg dest"] = 1.00
        parentConnectionExceptions["txThing2 txNamed narg source"] = 1.00
        parentConnectionExceptions["txThing2 txNonPred narg source"] = 0.81
        parentConnectionExceptions["txThing2 txPerson1 arg dest"] = 1.00
        parentConnectionExceptions["txThing2 txPred narg source"] = 0.86
        parentConnectionExceptions["txWhy txDateEntity narg source"] = 1.00
        parentConnectionExceptions["txWhy txHaveConcession arg dest"] = 1.00
        parentConnectionExceptions["txWhy txNonPred narg source"] = 0.75

    sourceType = 'parent'
    destType = 'parent'
    if rel[:3]=='ARG':
        if '%s %s %s %s' % (source,dest,'arg','source') in parentConnectionExceptions:
            sourceType = 'child'
        if '%s %s %s %s' % (source,dest,'arg','dest') in parentConnectionExceptions:
            destType = 'child'
    else:
        if '%s %s %s %s' % (source,dest,'narg','source') in parentConnectionExceptions:
            sourceType = 'child'
        if '%s %s %s %s' % (source,dest,'narg','dest') in parentConnectionExceptions:
            destType = 'child'
    return sourceType, destType    


def getConceptDetailedTranslationHashes(wdir='./AMR have-rel-role_etc'):

    fn = 'morph-verbalization-v1.01.txt'
    fn = 'verbalization-list-v1.01.txt'
    fn = 'have-rel-role-91-roles-v1.01.txt'
    fn = 'have-org-role-91-roles-v1.01.txt'

    allDirs = {}
    
    fn = 'morph-verbalization-v1.01.txt'
    dtype = 'morphVerbalization'
    allDirs[dtype] = {}
    for line in open(wdir + '/' + fn ):
        zz= line.split()
        t = [re.sub(r'\"', '', z ) for z in zz]
        allDirs[dtype][ t[3] ] = t
    
    # VERBALIZE abusive TO abuse-01
    fn = 'verbalization-list-v1.01.txt'
    dtype = 'verbalization'
    allDirs[dtype] = {}
    for line in open(wdir + '/' + fn ):
        zz= line.split()
        if zz[0]=='VERBALIZE':
            t = [re.sub(r'\"', '', z ) for z in zz]
            allDirs[dtype][ t[1] ] = t[2:]   # changed 
        
    # USE-HAVE-REL-ROLE-91-ARG2 enemy
    fn = 'have-rel-role-91-roles-v1.01.txt'
    dtype = 'relRole'
    allDirs[dtype] = {}
    for line in open(wdir + '/' + fn ):
        zz= line.split()
        if len(zz)==2:
            t = [re.sub(r'\"', '', z ) for z in zz]
            allDirs[dtype][ t[1] ] = t[0]
        
    # USE-HAVE-ORG-ROLE-91-ARG2 undersecretary-general
    fn = 'have-org-role-91-roles-v1.01.txt'
    dtype = 'orgRole'
    allDirs[dtype] = {}
    for line in open(wdir + '/' + fn ):
        zz= line.split()
        if zz[0]=='USE-HAVE-ORG-ROLE-91-ARG2':
            t = [re.sub(r'\"', '', z ) for z in zz]
            allDirs[dtype][ t[1] ] = t[0]
        
    return (allDirs) 
    
    

def getTxFuncs(nameWordIndices, tag):
    txFuncs=[]
    for i,ix in enumerate(nameWordIndices):
        if len(nameWordIndices)==1:
            txFuncs.append( (ix, 'S_'+tag)  )
        elif i==0:
            txFuncs.append( (ix, 'B_'+tag)  )
        elif i==(len(nameWordIndices)-1):
            txFuncs.append( (ix, 'E_'+tag)  )
        else:
            txFuncs.append( (ix, 'I_'+tag)  )
    return txFuncs        

def nodesUnassigned(conceptList, nodes):
    for n in nodes:
        if not n in conceptList:
            return False
    return True


def getNEWSentenceDFTagList():
    #  _p and _c for parent, child:    
    # ar0_arg  ar0_ix ar1_arg  ar1_ix  ar2_arg  ar2_ix  ar3_arg  ar3_ix 
    # attr0_lbl attr0_val attr1_lbl attr1_val  attr2_lbl  attr2_val  attr3_lbl  attr3_val  
    # nar0_ix nar0_lbl  nar1_ix nar1_lbl  nar2_ix  nar2_lbl  nar3_ix  nar3_lbl 
    # kind sense  
    #  add this
    # edge_p_c
    
    """
    sentIX    
    wordIX  
    words 
    txBIOES      
    
    # _p and _c for parent, child:    
    ar0_arg  ar0_ix ar1_arg  ar1_ix  ar2_arg  ar2_ix  ar3_arg  ar3_ix 
    attr0_lbl attr0_val attr1_lbl attr1_val  attr2_lbl  attr2_val  attr3_lbl  attr3_val  
    nar0_ix nar0_lbl  nar1_ix nar1_lbl  nar2_ix  nar2_lbl  nar3_ix  nar3_lbl 
    kind sense  
    
    # add this
    edge_p_c
    
    NcatProb pVectorL0  pVectorL0Args  pVectorL0Attr  pVectorL0Nargs txBIOESProb 
   
    # maybe the rest of these could go away    
    nameCategory  
    wiki           
    nnChild nnParent  
    txFunc           
    top    
    txBIOES_2 txBIOES_3 txBIOES_4 
    wordIX0  wordIX1  wordIX2  wordIX3  wordIX4  wordIX5  wordIX6  wordIX7          
    """

    tagList = [ 'sentIX', 'wordIX', 'words',  'txBIOES', 'txBIOESProb', 'NcatProb', 'sense',   ] # NEW 6/29/16 put sense back in
    
    for nodePostfix in ['_p', '_c']:
        for i in range(4): # max of four args per predicate
            tagList += [  'ar%d_arg%s' % (i,nodePostfix), 'ar%d_ix%s' % (i,nodePostfix), 'ar%d_destPC%s' % (i,nodePostfix)  ]
        for i in range(4): # max of four nonarg relations per nonPred
            tagList += [  'nar%d_lbl%s' % (i,nodePostfix), 'nar%d_ix%s' % (i,nodePostfix), 'nar%d_destPC%s' % (i,nodePostfix)  ]
        for i in range(4): # max of four attributes per concept
            tagList += [  'attr%d_lbl%s' % (i,nodePostfix), 'attr%d_val%s' % (i,nodePostfix)  ]
        tagList += ['kind'+nodePostfix ]
    
    tagList += ['edge_p_c']    
    tagList += ['pVectorL0',  'pVectorL0Args',  'pVectorL0Nargs',  'pVectorL0Attr']    
        
    # obsolete legacy, delete when safety is confirmed    
    tagList += ['txFunc', 'nameCategory', 'wiki', 'top', 'nnParent', 'nnChild']
    for i in range(8): # store all wordIX's
        tagList += [  'wordIX%d' % i   ]
    return tagList
    


def compareDictionaries( ref, sys ):
    compared = 0 
    refTotal = len(ref.keys())
    sysTotal = len(sys.keys())

    for key, val in ref.iteritems():
        if key in sys:
            if sys[key].lower() == val.lower():
                compared += 1
        
    return refTotal, sysTotal, compared

def getDateEntityAttributes(words):
    allWords = ' '.join(words)
    attribDict = {}
    year=0     
    month=0     
    day=0     
    if len(words) == 1 and words[0].isdigit():
        m = re.match(r'^(\d\d\d\d)$', allWords)
        if m:
            #print m.group(1), allWords
            attribDict['year'] = m.group(1)
            
        m = re.match(r'^(\d\d)(\d\d)(\d\d)$', allWords)
        if m:
            year  = m.group(1)
            month = m.group(2)
            day   = m.group(3)
            if int(year) < 40:
                year = '20'+year
            else:
                year = '19'+year
        m = re.match(r'^(\d\d\d\d)(\d\d)(\d\d)$', allWords)
        if m:
            year  = m.group(1)
            month = m.group(2)
            day   = m.group(3)
            #print year, month, day, allWords
        m = re.match(r'^(\d\d)$', allWords)
        if m:
            day   = m.group(1)
            #print year, month, day, allWords
        if year or month or day:
            if int(year)>0:
                attribDict['year'] = year
            if int(month)>0:
                attribDict['month'] = str(int(month))
            if int(day)>0:
                attribDict['day'] = str(int(day))
                
    if attribDict == {}: # 2008 - 09 - 26
        
        m = re.search(r'(\s+|^)(\d\d\d\d)\s+\D*\s+(\d\d)\s+\D*\s+(\d\d)(\s+|$)', allWords);            
        if m:
            year  = m.group(2)
            month = m.group(3)
            day   = m.group(4)
        if year or month or day:
            if int(year)>0:
                attribDict['year'] = year
            if int(month)>0:
                attribDict['month'] = str(int(month))
            if int(day)>0:
                attribDict['day'] = str(int(day))
         
    if attribDict == {}: # Named Month
        months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']            
        for mn in months:
            m = re.search(r'(?:\s+|^)(%s)' % mn, allWords, flags=re.IGNORECASE)
            if m:
                month = months.index(m.group(1).lower()) + 1        
        m = re.search(r'(?:\s+|^)(\d\d\d\d)(?:\s+|$)', allWords)
        if m:
           year = m.group(1)
        m = re.search(r'(?:\s+|^)(\d\d)(?:\s+|$)', allWords)
        if m:
           day = m.group(1)
        m = re.search(r'(?:\s+|^)(\d)(?:\s+|$)', allWords)
        if m:
           day = m.group(1)
        if year or month or day:
            if year and int(year)>0:
                attribDict['year'] = year
            if month and int(month)>0:
                attribDict['month'] = str(int(month))
            if day and int(day)>0:
                attribDict['day'] = str(int(day))
   
    return attribDict    

def text2int(textnum, numwords={}):
    # from http://stackoverflow.com/questions/493174/is-there-a-way-to-convert-number-words-to-integers
    found=False
    textnum = textnum.lower()
    if not numwords:
      units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
      ]

      tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

      scales = ["hundred", "thousand", "million", "billion", "trillion"]

      numwords["and"] = (1, 0)
      for idx, word in enumerate(units):    numwords[word] = (1, idx)
      for idx, word in enumerate(tens):     numwords[word] = (1, idx * 10)
      for idx, word in enumerate(scales):   numwords[word] = (10 ** (idx * 3 or 2), 0)

    current = result = 0
    for word in textnum.split():
        if word in numwords:
            scale, increment = numwords[word]
            if (scale > 100) and not found:
                current = 1
            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0
            found=True
    if not found:
        if 'hundreds'  in textnum : return 1e2
        if 'thousands' in textnum : return 1e3
        if 'millions'  in textnum : return 1e6
        if 'billions'  in textnum : return 1e9
        if 'trillions' in textnum : return 1e12

        return None
    else:
        return result + current

def getTemporalQuantityQuantUnit(words):
    """
    from training data:
year        392  56.402878      392   56.402878
day         115  16.546763      507   72.949640
month        70  10.071942      577   83.021583
week         36   5.179856      613   88.201439
hour         35   5.035971      648   93.237410
decade       22   3.165468      670   96.402878
minute       16   2.302158      686   98.705036
century       4   0.575540      690   99.280576
second        4   0.575540      694   99.856115
    """
    allWords = ' '.join(words)
    
    unitKind = 'year'
    quant = None
    units = ['year', 'day', 'month', 'week', 'hour', 'decade', 'minute', 'century', 'second']
    for u in units:
        if u in allWords : unitKind=u; break
    if 'centuries' in allWords : unitKind='century'
    if 'daily' in allWords : unitKind='day'
    m = re.search(r'([0-9,\.]+)', allWords.lower())
    if m:
        quant = m.group(1)
        quant = quant.replace(",", "")
        if quant=='.':
            quant=None
    if not quant:
        q = text2int(allWords)
        if q:
            quant = str(q)
    if not quant:
        quant = '1'
    

    return   quant, unitKind    
        
def getMonetaryQuantityQuantUnit(words):
    """
    from training data:
          count       perc  cum_sum    cum_perc
kind_c                                         
dollar      133  60.454545      133   60.454545
yuan         36  16.363636      169   76.818182
euro         23  10.454545      192   87.272727
pound         7   3.181818      199   90.454545
cent          5   2.272727      204   92.727273
rupee         3   1.363636      207   94.090909
renminbi      2   0.909091      209   95.000000
diram         1   0.454545      210   95.454545
figure        1   0.454545      211   95.909091
franc         1   0.454545      212   96.363636
mark          1   0.454545      213   96.818182
baht          1   0.454545      214   97.272727
quid          1   0.454545      215   97.727273
ruble         1   0.454545      216   98.181818
somoni        1   0.454545      217   98.636364
won           1   0.454545      218   99.090909
yen           1   0.454545      219   99.545455
RMB           1   0.454545      220  100.000000
    """
    allWords = ' '.join(words)
    
    unitKind = 'dollar'
    quant = None
    units = ['dollar', 'yuan', 'euro', 'pound', 'cent', 'rupee', 'franc', 'mark', 'yen']
    for u in units:
        if u in allWords : unitKind=u; break
    if 'centuries' in allWords : unitKind='century'
    if 'daily' in allWords : unitKind='day'
    m = re.search(r'([0-9,\.]+)', allWords.lower())
    if m:
        quant = m.group(1)
        quant = quant.replace(",", "")
        quant = float(quant)
        if quant=='.':
            quant=None
    if not quant:
        q = text2int(allWords)
        if q:
            quant = q
    else:
        m = text2int(allWords)
        if m:
            quant *= m
    if not quant:
        quant = 1
    quant = ('%f' % quant).rstrip('0').rstrip('.')
        

    return   quant, unitKind    
        
def getMassQuantityQuantUnit(words):
    """
    from training data:
          count       perc  cum_sum    cum_perc
kind_c                                         
kilogram     66  43.421053       66   43.421053
ton          45  29.605263      111   73.026316
gram         24  15.789474      135   88.815789
pound        11   7.236842      146   96.052632
kilo          3   1.973684      149   98.026316
ounce         1   0.657895      150   98.684211
picogram      1   0.657895      151   99.342105
tonne         1   0.657895      152  100.000000
    """
    allWords = ' '.join(words)
    
    unitKind = 'kilogram'
    quant = None
    units = ['kilogram', 'tonne', 'ton', 'picogram', 'gram', 'pound',  'ounce']
    for u in units:
        if u in allWords : unitKind=u; break
    if 'kilo' in allWords : unitKind='kilogram'
    if 'daily' in allWords : unitKind='day'
    m = re.search(r'([0-9,\.]+)', allWords.lower())
    if m:
        quant = m.group(1)
        quant = quant.replace(",", "")
        quant = float(quant)
        if quant=='.':
            quant=None
    if not quant:
        q = text2int(allWords)
        if q:
            quant = q
    else:
        m = text2int(allWords)
        if m:
            quant *= m
    if not quant:
        quant = 1
    quant = ('%f' % quant).rstrip('0').rstrip('.')
        

    return   quant, unitKind    

def getDistanceQuantityQuantUnit(words):
    """
    from training data:
kind_c                                           
kilometer      31  37.349398       31   37.349398
mile           23  27.710843       54   65.060241
meter           9  10.843373       63   75.903614
foot            8   9.638554       71   85.542169
inch            7   8.433735       78   93.975904
millimeter      2   2.409639       80   96.385542
nanometer       1   1.204819       81   97.590361
step            1   1.204819       82   98.795181
street          1   1.204819       83  100.000000
    """
    allWords = ' '.join(words)
    
    unitKind = 'kilometer'
    quant = None
    units = ['kilometer', 'mile', 'millimeter', 'nanometer', 'meter', 'foot', 'inch', 'step', 'street']
    for u in units:
        if u in allWords : unitKind=u; break
    if 'feet' in allWords : unitKind='foot'
    m = re.search(r'([0-9,\.]+)', allWords.lower())
    if m:
        quant = m.group(1)
        quant = quant.replace(",", "")
        quant = float(quant)
        if quant=='.':
            quant=None
    if not quant:
        q = text2int(allWords)
        if q:
            quant = q
    else:
        m = text2int(allWords)
        if m:
            quant *= m
    if not quant:
        quant = 1
    quant = ('%f' % quant).rstrip('0').rstrip('.')
        

    return   quant, unitKind    
        
def getAreaQuantityQuantUnit(words):
    """
    from training data:
              count       perc  cum_sum    cum_perc
kind_c                                             
hectare           7  58.333333        7   58.333333
acre              2  16.666667        9   75.000000
meter             1   8.333333       10   83.333333
square-foot       1   8.333333       11   91.666667
square-meter      1   8.333333       12  100.000000
    """
    allWords = ' '.join(words)
    
    unitKind = 'hectare'
    quant = None
    units = ['hectare', 'acre', 'euro', 'meter', 'square-foot', 'square-meter' ]
    for u in units:
        if u in allWords : unitKind=u; break
    if 'square foot' in allWords : unitKind='square-foot'
    if 'square feet' in allWords : unitKind='square-foot'
    if 'square meter' in allWords : unitKind='square-meter'
    m = re.search(r'([0-9,\.]+)', allWords.lower())
    if m:
        quant = m.group(1)
        quant = quant.replace(",", "")
        quant = float(quant)
        if quant=='.':
            quant=None
    if not quant:
        q = text2int(allWords)
        if q:
            quant = q
    else:
        m = text2int(allWords)
        if m:
            quant *= m
    if not quant:
        quant = 1
    quant = ('%f' % quant).rstrip('0').rstrip('.')
        

    return   quant, unitKind    
        
        
        #subGraph['attrDict_p'] =  attrDict_p

def updateSubGraphStats(tx, words, refSubGraph, subGraph, subGraphStats):
    if not tx in subGraphStats:
        subGraphStats[tx] = {}
    # subGraphStats[tx][words][subGraph['kind_p']] += 1
    # print '%10s %30s: %s' % (tx, words, refSubGraph)
    # print '%10s %30s: %s' % ('', '', subGraph)


[senseDict, allWordStats, singleWordStats] = pickle.load(  open( 'conceptDicts.pcl', "rb" ) )         
cdtHash = getConceptDetailedTranslationHashes()  # do this lazily, like other hash list loading




def getMostCommonTranslation(tx, words, k):
    if tx in allWordStats:
        if k in allWordStats[tx]:
            allWords = ' '.join(words).lower()
            if allWords in allWordStats[tx][k]:
                if len(allWordStats[tx][k][allWords]) > 1:
                    mxCount=0
                    mxItem=''
                    for item in allWordStats[tx][k][allWords]:
                        if allWordStats[tx][k][allWords][item] > mxCount:
                            mxCount = allWordStats[tx][k][allWords][item]
                            mxItem  = item
                    return mxItem 
                else:
                    return allWordStats[tx][k][allWords].keys()[0]
            if words[0] in allWordStats[tx][k]:
                if len(allWordStats[tx][k][words[0]]) > 1:
                    mxCount=0
                    mxItem=''
                    for item in allWordStats[tx][k][words[0]]:
                        if allWordStats[tx][k][words[0]][item] > mxCount:
                            mxCount = allWordStats[tx][k][words[0]][item]
                            mxItem  = item
                    return mxItem 
                else:
                    return allWordStats[tx][k][words[0]].keys()[0]
        lemma=lmtzr.lemmatize(words[0].lower())
        if tx == 'txPred':
            if words[0] in senseDict:
                sense = senseDict[words[0]]    
            else:    
                sense = '01'    
            lemma=lmtzr.lemmatize(words[0].lower(), 'v')
            predWithSense = lemma
            if (sense):
                predWithSense = lemma + '-' + sense
            return predWithSense
        elif tx == 'txNamed':
            return 'person'
        else:
            return TXConvert.convert(lemma)   
    return None            

def getSubGraph(txBIOES, words, wkLink='-'):  
    
    subGraph = { 'kind_p':None, 'attrDict_p':{},  'kind_c':None, 'attrDict_c':{}, 'edge_p_c':None }
    
    if txBIOES=='txNonPred':
        #lemma=lmtzr.lemmatize(words[0].lower())
        #subGraph['kind_p'] = TXConvert.convert(lemma)
        #print 'txNonPred DEBUG', words[0].lower(), lemma, TXConvert.convert(lemma)
        subGraph['kind_p'] = getMostCommonTranslation(txBIOES, words, 'kind_p')

    if (txBIOES=='txPred-01'):
        if words[0] in senseDict:
            sense = senseDict[words[0]]    
        else:    
            sense = '01'    
        lemma=lmtzr.lemmatize(words[0].lower(), 'v')
        predWithSense = lemma
        if (sense):
            predWithSense = lemma + '-' + sense
        subGraph['kind_p'] = TXConvert.convert(predWithSense)  # dominance-01 => dominate-01  (sense affects lemma)
        alternate = getMostCommonTranslation(txBIOES, words, 'kind_p')
        if alternate:
            subGraph['kind_p'] = alternate
    
    if (txBIOES=='txPred-02'):
        sense='02'
        #if words[0] in senseDict:
        #    sense = senseDict[words[0]]    
        #else:    
        #    sense = '01'    
        lemma=lmtzr.lemmatize(words[0].lower(), 'v')
        predWithSense = lemma
        if (sense):
            predWithSense = lemma + '-' + sense
        subGraph['kind_p'] = TXConvert.convert(predWithSense)  # dominance-01 => dominate-01  (sense affects lemma)
        alternate = getMostCommonTranslation(txBIOES, words, 'kind_p')
        if alternate:
            subGraph['kind_p'] = alternate
    
    if (txBIOES=='txPred'):
        if words[0] in senseDict:
            sense = senseDict[words[0]]    
        else:    
            sense = '01'    
        lemma=lmtzr.lemmatize(words[0].lower(), 'v')
        predWithSense = lemma
        if (sense):
            predWithSense = lemma + '-' + sense
        subGraph['kind_p'] = TXConvert.convert(predWithSense)  # dominance-01 => dominate-01  (sense affects lemma)
        alternate = getMostCommonTranslation(txBIOES, words, 'kind_p')
        if alternate:
            subGraph['kind_p'] = alternate
    
    if (txBIOES=='txNamed'):
        # Need words, op, category, wiki for each wikification
        subGraph['kind_p'] = getMostCommonTranslation(txBIOES, words, 'kind_p')
        subGraph['kind_c'] =  'name'
        subGraph['edge_p_c'] =  'name'
        attrDict_p={}
        attrDict_p['wiki'] = wkLink # '"' + wkLink + '"' 
        attrDict_c={}
        for ai in range(len(words)):
            lbl = 'op%d' % (ai+1) 
            val = words[ai].lower()
            attrDict_c[lbl] = val
            
        subGraph['attrDict_p'] =  attrDict_p
        subGraph['attrDict_c'] =  attrDict_c




    if (txBIOES=='txDateEntity'):  # get this from the stanford parser
        #dt = dateparser.parse(words.join())
        #print words, dt
        # date-entity         nan         nan                                        {}  {u'year': u'2008', u'day': u'24', u'month': u'8'} 
        subGraph['kind_p']      =  'date-entity'
        subGraph['attrDict_p']  = getDateEntityAttributes(words)

    if (txBIOES=='txHaveOrg'):
        # txHaveOrg                     have-org-role-91        ARG2    official                                        {}  {} 

        
        """
        
        some txHaveOrg don't have three elements?
        lots of child kind miscompares (lower case, singularization)
                txFunc  count  refCount  sysCount  goodCount  precision    recall        f1  kind_p_acc  kind_c_acc  edge_p_c_acc  attrDict_p_acc  attrDict_c_acc  sense_acc  lemma_acc
4            txHaveOrg   1742      5187      5226       2358   0.451206  0.454598  0.452895    1.000000    0.363794             1        0.000000             NaN        NaN        NaN     
4            txHaveOrg   1742      5187      1742       1729   0.992537  0.333333  0.499062    1.000000           0             0        0.000000             NaN        NaN        NaN
        """
        
        subGraph['kind_p'] = 'have-org-role-91'
        #subGraph['kind_c'] = words[0]
        subGraph['kind_c'] = lmtzr.lemmatize(words[0].lower(), 'n').lower()
        subGraph['edge_p_c'] = 'ARG2'

    if (txBIOES=='txPerson0'):
        # txPerson0                     abuse-02        ARG0      person                                        {}  {} 
        # morph-verbalization-v1.01.txt:::DERIV-VERB "decide" ::DERIV-NOUN "decision"

        k = 'verbalization'
        w = words[0].lower()
        w=lmtzr.lemmatize(w.lower())
        if w in cdtHash['morphVerbalization']:
            #print ' ========> %24s  %20s -> %20s' % ('morphVerbalization', w, cdtHash['morphVerbalization'][w])
            w = cdtHash['morphVerbalization'][w][1]     
        if w in cdtHash['verbalization']:
            #print ' ========> %24s  %20s -> %20s' % ('verbalization', w, cdtHash['verbalization'][w])
            predWithSense = cdtHash['verbalization'][w][-1]
        else:
            #print '  NOT FOUND', w
            lemma=lmtzr.lemmatize(words[0].lower(), 'v')
            sense = '01'
            predWithSense = lemma
            if (sense):
                predWithSense = lemma + '-' + sense
        subGraph['kind_p'] = TXConvert.convert(predWithSense)  
        subGraph['kind_c'] = 'person'
        subGraph['edge_p_c'] = 'ARG0'

    if (txBIOES=='txFrequentDouble'):
        # txFrequentDouble              govern-01        ARG0  government-organization                                        {}  {} 
        subGraph['kind_p'] = 'govern-01'
        subGraph['kind_c'] = 'government-organization'
        subGraph['edge_p_c'] = 'ARG0'

    if (txBIOES=='txTemporalQuantity'):
        # txTemporalQuantity            temporal-quantity        unit         day                                        {}  {u'quant': u'1'} 
        subGraph['kind_p']   = 'temporal-quantity'
        subGraph['edge_p_c'] = 'unit'
        quant, unitKind    = getTemporalQuantityQuantUnit(words)
        subGraph['attrDict_p']  = {'quant': quant}
        subGraph['kind_c']      =  unitKind



    if (txBIOES=='txThing1'):
        # txThing1                      find-01        ARG1       thing                                        {}  {} 
        k = 'verbalization'
        w = words[0].lower()
        w=lmtzr.lemmatize(w.lower())
        if w in cdtHash['morphVerbalization']:
            #print ' ========> %24s  %20s -> %20s' % ('morphVerbalization', w, cdtHash['morphVerbalization'][w])
            w = cdtHash['morphVerbalization'][w][1]     
        if w in cdtHash['verbalization']:
            #print ' ========> %24s  %20s -> %20s' % ('verbalization', w, cdtHash['verbalization'][w])
            predWithSense = cdtHash['verbalization'][w][-1]
        else:
            #print '  NOT FOUND', w
            lemma=lmtzr.lemmatize(words[0].lower(), 'v')
            sense = '01'
            predWithSense = lemma
            if (sense):
                predWithSense = lemma + '-' + sense
        subGraph['kind_p'] = TXConvert.convert(predWithSense)  
        subGraph['kind_c'] = 'thing'
        subGraph['edge_p_c'] = 'ARG1'


    if (txBIOES=='txPerson1'):
        # txPerson1                     vote-01        ARG1      person                                        {}  {} 

        w = words[0].lower()
        w=lmtzr.lemmatize(w.lower())
        if w in cdtHash['morphVerbalization']:
            #print ' ========> %24s  %20s -> %20s' % ('morphVerbalization', w, cdtHash['morphVerbalization'][w])
            w = cdtHash['morphVerbalization'][w][1]     
        if w in cdtHash['verbalization']:
            #print ' ========> %24s  %20s -> %20s' % ('verbalization', w, cdtHash['verbalization'][w])
            predWithSense = cdtHash['verbalization'][w][-1]
        else:
            #print '  NOT FOUND', w
            lemma=lmtzr.lemmatize(words[0].lower(), 'v')
            sense = '01'
            predWithSense = lemma
            if (sense):
                predWithSense = lemma + '-' + sense
        subGraph['kind_p'] = TXConvert.convert(predWithSense)  
        subGraph['kind_c'] = 'person'
        subGraph['edge_p_c'] = 'ARG1'

    if (txBIOES=='txMore'):
        # txMore                        late      degree        more                                        {}  {} 
        subGraph['kind_p'] = getMostCommonTranslation(txBIOES, words, 'kind_p')
        subGraph['kind_c'] = 'more'
        subGraph['edge_p_c'] = 'degree'

    if (txBIOES=='txInclude'):
        # txInclude                     include-91        ARG2       power                                        {}  {} 
        w = words[0].lower()
        w=lmtzr.lemmatize(w.lower())
        subGraph['kind_p'] = 'include-91'
        subGraph['kind_c'] = w
        subGraph['edge_p_c'] = 'ARG2'

    if (txBIOES=='txThing2'):
        # txThing2                      use-01        ARG2       thing                                        {}  {} 
        # txThing1                      find-01        ARG1       thing                                        {}  {} 
        k = 'verbalization'
        w = words[0].lower()
        w=lmtzr.lemmatize(w.lower())
        if w in cdtHash['morphVerbalization']:
            #print ' ========> %24s  %20s -> %20s' % ('morphVerbalization', w, cdtHash['morphVerbalization'][w])
            w = cdtHash['morphVerbalization'][w][1]     
        if w in cdtHash['verbalization']:
            #print ' ========> %24s  %20s -> %20s' % ('verbalization', w, cdtHash['verbalization'][w])
            predWithSense = cdtHash['verbalization'][w][-1]
        else:
            #print '  NOT FOUND', w
            lemma=lmtzr.lemmatize(words[0].lower(), 'v')
            sense = '01'
            predWithSense = lemma
            if (sense):
                predWithSense = lemma + '-' + sense
        subGraph['kind_p'] = TXConvert.convert(predWithSense)  
        subGraph['kind_c'] = 'thing'
        subGraph['edge_p_c'] = 'ARG2'

    if (txBIOES=='txHaveRel'):
        # txHaveRel                     have-rel-role-91        ARG2       enemy                                        {}  {} 
        subGraph['kind_p'] = 'have-rel-role-91'
        w = words[0].lower()
        w=lmtzr.lemmatize(w.lower())
        subGraph['kind_p'] = 'have-rel-role-91'
        subGraph['kind_c'] = w
        subGraph['edge_p_c'] = 'ARG2'

    if (txBIOES=='txMost'):
        # txMost                        long-03      degree        most                                        {}  {} 
        subGraph['kind_p'] = getMostCommonTranslation(txBIOES, words, 'kind_p')
        subGraph['kind_c']   = 'most'
        subGraph['edge_p_c'] = 'degree'

    if (txBIOES=='txPerson2'):
        # txPerson1                     vote-01        ARG1      person                                        {}  {} 

        k = 'verbalization'
        w = words[0].lower()
        w=lmtzr.lemmatize(w.lower())
        if w in cdtHash['morphVerbalization']:
            #print ' ========> %24s  %20s -> %20s' % ('morphVerbalization', w, cdtHash['morphVerbalization'][w])
            w = cdtHash['morphVerbalization'][w][1]     
        if w in cdtHash['verbalization']:
            #print ' ========> %24s  %20s -> %20s' % ('verbalization', w, cdtHash['verbalization'][w])
            predWithSense = cdtHash['verbalization'][w][-1]
        else:
            #print '  NOT FOUND', w
            lemma=lmtzr.lemmatize(words[0].lower(), 'v')
            sense = '01'
            predWithSense = lemma
            if (sense):
                predWithSense = lemma + '-' + sense
        subGraph['kind_p'] = TXConvert.convert(predWithSense)  
        subGraph['kind_c'] = 'person'
        subGraph['edge_p_c'] = 'ARG2'

    if (txBIOES=='txThing0'):
        # txThing0                      evidence-01        ARG0       thing                                        {}  {} 
        # txThing1                      find-01        ARG1       thing                                        {}  {} 
        k = 'verbalization'
        w = words[0].lower()
        w=lmtzr.lemmatize(w.lower())
        if w in cdtHash['morphVerbalization']:
            #print ' ========> %24s  %20s -> %20s' % ('morphVerbalization', w, cdtHash['morphVerbalization'][w])
            w = cdtHash['morphVerbalization'][w][1]     
        if w in cdtHash['verbalization']:
            #print ' ========> %24s  %20s -> %20s' % ('verbalization', w, cdtHash['verbalization'][w])
            predWithSense = cdtHash['verbalization'][w][-1]
        else:
            #print '  NOT FOUND', w
            lemma=lmtzr.lemmatize(words[0].lower(), 'v')
            sense = '01'
            predWithSense = lemma
            if (sense):
                predWithSense = lemma + '-' + sense
        subGraph['kind_p'] = TXConvert.convert(predWithSense)  
        subGraph['kind_c'] = 'thing'
        subGraph['edge_p_c'] = 'ARG0'

    if (txBIOES=='txHaveConcession'):
        # txHaveConcession              have-concession-91        ARG2          or                                        {}  {'TOP': u'have-concession-91'} 
        subGraph['kind_p'] = 'have-concession-91'
        subGraph['kind_c'] = getMostCommonTranslation(txBIOES, words, 'kind_c')
        subGraph['edge_p_c'] = 'ARG2'

    if (txBIOES=='txMonetaryQuantity'):
        # txMonetaryQuantity            monetary-quantity        unit      dollar                                        {}  {u'quant': u'295000000'} 
        subGraph['kind_p'] = 'monetary-quantity'
        subGraph['edge_p_c'] = 'unit'
        quant, unitKind    = getMonetaryQuantityQuantUnit(words)
        subGraph['attrDict_p']  = {'quant': quant}
        subGraph['kind_c']      =  unitKind

    if (txBIOES=='txMassQuantity'):
        # txMassQuantity                mass-quantity        unit         ton                                        {}  {u'quant': u'1.6'} 
        subGraph['kind_p'] = 'mass-quantity'
        subGraph['edge_p_c'] = 'unit'
        quant, unitKind    = getMassQuantityQuantUnit(words)
        subGraph['attrDict_p']  = {'quant': quant}
        subGraph['kind_c']      =  unitKind

    if (txBIOES=='txAgo'):
        # txAgo                         before         op1         now                                        {}  {} 
        subGraph['kind_p'] = 'before'
        subGraph['kind_c'] = 'now'
        subGraph['edge_p_c'] = 'op1'

    if (txBIOES=='txDistanceQuantity'):
        # txDistanceQuantity            distance-quantity        unit   kilometer                                        {}  {u'quant': u'76'} 
        subGraph['kind_p'] = 'distance-quantity'
        subGraph['edge_p_c'] = 'unit'
        quant, unitKind    = getDistanceQuantityQuantUnit(words)
        subGraph['attrDict_p']  = {'quant': quant}
        subGraph['kind_c']      =  unitKind

    if (txBIOES=='txAreaQuantity'):
        # txDistanceQuantity            distance-quantity        unit   kilometer                                        {}  {u'quant': u'76'} 
        subGraph['kind_p'] = 'area-quantity'
        subGraph['edge_p_c'] = 'unit'
        quant, unitKind    = getAreaQuantityQuantUnit(words)
        subGraph['attrDict_p']  = {'quant': quant}
        subGraph['kind_c']      =  unitKind

    if (txBIOES=='txWhy'):
        # txWhy                         cause-01        ARG0  amr-unknown                                        {}  {} 
        subGraph['kind_p'] = 'cause-01'
        subGraph['kind_c'] = 'amr-unknown'
        subGraph['edge_p_c'] = 'ARG0'

    if (txBIOES=='txHaveCondition'):
        # txHaveCondition               have-condition-91        ARG2       go-06                                        {}  {} 
        subGraph['kind_p'] = 'have-condition-91'
        subGraph['kind_c'] = getMostCommonTranslation(txBIOES, words, 'kind_c')
        subGraph['edge_p_c'] = 'ARG2'

    if (txBIOES=='txBeLocatedAt'):
        # txBeLocatedAt                 be-located-at-91        ARG2        edge                                        {}  {'TOP': u'be-located-at-91'} 
        w = words[0].lower()
        w=lmtzr.lemmatize(w.lower())
        subGraph['kind_p'] = 'be-located-at-91'
        subGraph['kind_c'] = w
        subGraph['edge_p_c'] = 'ARG2'

    if (txBIOES=='txInsteadOf'):
        # txInsteadOf                   instead-of-91        ARG2     missile                                        {}  {} 
        w = words[0].lower()
        w=lmtzr.lemmatize(w.lower())
        subGraph['kind_p'] = 'instead-of-91'
        subGraph['kind_c'] = getMostCommonTranslation(txBIOES, words, 'kind_c')
        subGraph['edge_p_c'] = 'ARG2'

    if (txBIOES=='txHavePart'):
        # txHaveParthave-part-91        ARG2        that                                        {}  {'TOP': u'have-part-91'} 
        w = words[0].lower()
        w=lmtzr.lemmatize(w.lower())
        subGraph['kind_p'] = 'have-part-91'
        subGraph['kind_c'] = w
        subGraph['edge_p_c'] = 'ARG2'

    if (txBIOES=='txBeTemporallyAt'):
        # txBeTemporallyAt              be-temporally-at-91        ARG2        week                                        {}  {'TOP': u'be-temporally-at-91'} 
        w = words[0].lower()
        w=lmtzr.lemmatize(w.lower())
        subGraph['kind_p'] = 'be-temporally-at-91'
        subGraph['kind_c'] = w
        subGraph['edge_p_c'] = 'ARG2'

    if (txBIOES=='txHavePurpose'):
        # txHavePurpose                 have-purpose-91        ARG2         and                                        {}  {'TOP': u'have-purpose-91'} 
        w = words[0].lower()
        w=lmtzr.lemmatize(w.lower())
        subGraph['kind_p'] = 'have-purpose-91'
        subGraph['kind_c'] = w
        subGraph['edge_p_c'] = 'ARG2'

    if (txBIOES=='txRateEntity'):
        # txRateEntity                  nan         nan         nan                                        {}  {} 
        w = words[0].lower()
        w=lmtzr.lemmatize(w.lower())
        subGraph['kind_p'] = 'rate-entity-91'
        subGraph['kind_c'] = w
        subGraph['edge_p_c'] = 'ARG2'

    if (txBIOES=='txVolumeQuantity'):
        # txVolumeQuantity              volume-quantity        unit         ton                                        {}  {u'quant': u'200'} 
        w = words[0].lower()
        w=lmtzr.lemmatize(w.lower())
        subGraph['kind_p'] = 'volume-quantity'
        subGraph['kind_c'] = w
        subGraph['edge_p_c'] = 'unit'

    if (txBIOES=='txHaveSubevent'):
        # txHaveSubevent                have-subevent-91        ARG2    violence                                        {}  {} 
        w = words[0].lower()
        w=lmtzr.lemmatize(w.lower())
        subGraph['kind_p'] = 'have-subevent-91'
        subGraph['kind_c'] = w
        subGraph['edge_p_c'] = 'ARG2'

    if (txBIOES=='txAnd'):
        # txHaveSubevent                have-subevent-91        ARG2    violence                                        {}  {} 
        subGraph['kind_p'] = 'and'

    if not subGraph['kind_p']:
        subGraph['kind_p'] = txBIOES   # at least assign something to this sub graph.
    
    return subGraph









if __name__ == '__main__':
    
    exit(1)
