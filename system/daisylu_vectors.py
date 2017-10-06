# 12/9/16, made this AMR specific, eliminating the need to parse srl vectors.
#
# DaisyLu Vectors  10/8/2015
#
# Read in the original conll format files and the list of items represented by each feature.
# Create MySQLite database with everything necessary to generate vectors for specific models,
# in Lua, including:
#    Word, Pred, PredContext(3), marker(3)  <-- first task, check with LUA generated SQLITE db which
#                                               will have differing indices for odd words, but not enough
#                                               to matter - ?  Then check the time for LUA vector creation
#                                               (should more processing be done here to speed up LUA?)x
#    Word, Pred, PredContext(5), marker(5) 
#    Word, Pred, PredContext-Pred(5), marker(5) 
#    Word, Pred, PredContext-Pred(5), Position, Caps, Suffix 
#    Word, Pred, PredContext-Pred(5), Position, Caps, Suffix, patch, deprel, POS, PPOS
#        
#
# Will extend this to new word reps, and to AMR feature generation, so be ready.  

#
# Similar to SRLDepPaper.py:  
# --------------------------
#
# It reads in CONLL format and generates a new CONLL format that includes path information 
#
# It generates a word list for random training that will only contain words from the training
# dataset, with a fraction removed in order to train the UNKNOWN vector.  This prevents using
# untrained, random words during testing.  
#
# It can be used to read in CONLL result files from the original contest, or from Daisy, and can 
# calculate the F1 score for them.
#
# It can separate the sense and role scores in many ways since the raw comparison data is stored
# in internal data structures.  It can rank systems by role and sense scores.
#
# It can read in the output from the official conll2009 perl script so that F1 can be compared.
#
# It can generate Latex tables of the results - PANDAS is better for this, though.
# 
# For the sense calculation, it evaluates senses for verbs and creates the list of preds that
# should be tested (versus preds that have the same value always in the test set)
#
# can create heatmaps, and plots of results for various feature combinations, used to generate
# comparative plots in dependency paper.
#
#

import sys
##reload(sys)
##sys.setdefaultencoding('utf-8')

import operator
import re
from pprint import pprint
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import platform
import sqlite3


class WordRepsFileLocations:
    # '../data/senna_embeddings.txt'
    
    pWordList     = '../data/senna_words.lst'    
    pWordRepsList = 'dummy.lst'    
    width = 302
    
    @staticmethod
    def init(pathToWordList):
        WordRepsFileLocations.pWordList     = pathToWordList
      
    @staticmethod
    def pathToWordList():
        return WordRepsFileLocations.pWordList

    @staticmethod
    def pathToWordRepsList():
        return WordRepsFileLocations.pWordRepsList

    @staticmethod
    def wordRepsWidth():
        return WordRepsFileLocations.width

            

def DBToDF(dbFn, tables= None):
    dFrames = {}
    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///' + dbFn, echo=False)
    if not tables:
        tables = engine.table_names() # read em all
    for t in tables:
        dFrames[t] = pd.read_sql_table(t, engine)
    return dFrames

def getTargetsFromModels(mi):
    features  = {}
    for tag in mi.keys():
        features[tag] = {}
        if mi[tag]['id'] > 0 :
            dbfn = '../data/' + mi[tag]['db']
            dfVec = DBToDF(dbfn, tables=['Tokens', 'GeneralArch'])
            target = dfVec['GeneralArch'][dfVec['GeneralArch']['key']=='target']['value'].tolist()[0]
            z = dfVec['Tokens'] 
            for fType in [target]:
                features[tag][fType] = dict(zip(z[(z['type']==fType)  ]['ix'].tolist(), z[(z['type']==fType)  ]['token'].tolist() ))
    return features
                                 
def parseFile(fn, sents):
    with open(fn) as f:
        s=[]
        for line in f:
            t = line.split()
            if (len(t) == 0):
                sents.append(s)
                s=[]
            else:
                s.append(t)

        
def parseWord(word, tokens):
    AMRCorpusTranslate={ '@-@':'-',  '@/@':'/',  '."':'.',  '",':',',   '".':'.',   '@:@':':',    
                         '@-':'-',   ')?':'?',   '!!':'!',  '??':'?',   '"?':'?',   '):':')'   }   
    if word in AMRCorpusTranslate:
        word =  AMRCorpusTranslate[word]
    lc = word.lower()
    #print   lc,
    if lc in tokens:
        return tokens[lc]
    if re.match("^\-?[\.\,\d]+$", lc):
        return tokens['0']
    else:
        m = re.search('^\'\d+$', lc)
        if m:
            t = "UNKNOWN"
            if t in tokens:
                return tokens[t]
        m = re.search('^\d+(-)\d+$', lc)
        if m:
            t = "00"
            if t in tokens:
                return tokens[t]
        m = re.search('^\d+(\D+)$', lc)
        if m:
            suffix = m.group(1)   
            t =  "0" + suffix
            if t in tokens:
                return tokens[t]
        m = re.search('^(\D+)\d+$', lc)
        if m:
            prefix = m.group(1)   
            t =  prefix + "0"
            if t in tokens:
                return tokens[t]
        m = re.search('^\d+(\D+)\d+$', lc)
        if m:
            mid = m.group(1)   
            t =  "0" + mid + "0"
            if t in tokens:
                return tokens[t]
    return tokens["UNKNOWN"]




def strListToCSV(a):
    st = u''
    if isinstance(a[0], list): # 2d 
        for row in a:
            st += ','.join([x.decode("utf8").encode('utf-8').strip() for x in row]) + '#'
    else:
        for i in range(len(a)):
            if isinstance(a[i], unicode):
                a[i] = a[i].encode('ascii', 'ignore').strip().encode('utf8')
           
        st += u','.join([x for x in a]) 
    return st


def intCSVToList(a):
    z = a.split(',')
    fz = [int(x) for x in z]
    return fz

def floatCSVToList(a):
    fList= []
    zz = a.split('#')
    for outer in zz:
        if outer=='': 
            continue
        z = outer.split(',')
        fz = [float(x) for x in z]
        fList.append(fz)
    return fList

def listToCSV(a):
    st = u''
    if isinstance(a[0], list): # 2d 
        for row in a:
            st += ','.join([str(x) for x in row]) + '#'
    else:
        st += ','.join([str(x) for x in a]) 
    return st

def strToList(st):
    if (st.find('#') > -1):
        twoD = []
        for row in st.strip('\0,#').split('#'):
            z = row.strip('\0,').split(',')
            twoD.append( [int(a) for a in z] )
        return twoD
    else:
        return [int(a) for a in str.rstrip('\0').split(',')]

def AddToAll(a, addend):
    if isinstance(a[0], list): # 2d 
        b=[]
        for row in a:
            b.append(  [x+addend for x in row])
        return b
    else:
        b = [x+addend for x in a]
        return b

def getFDef(fn):
    f={}
    f['tokens'] = [line.strip() for line in open(fn)]
    f['t2i']    = dict( [(x, y) for y, x in enumerate(f['tokens'])] )
    return f

def getDBPredicateCount(db):
    db.execute("SELECT COUNT(*) FROM Predicates")
    (ret) = db.fetchone()
    return ret[0]

def getDBPredicateVector(db, pnum):

    #db.execute("select ix, wfi, pfi, stTarget, ref_si, ref_pi from Predicates WHERE ix = ?", (pnum ))
    db.execute("select ix, wfi, pfi, stTarget, ref_si, ref_pi from Predicates WHERE ix = ?", (pnum, ))
    (_, _, pfi, stTarget, ref_si, ref_pi) = db.fetchone()
    ref_si -= 1 # Lua index
    ref_pi -= 1 # Lua index
    targetIndices = AddToAll(strToList(stTarget), -1)

    db.execute("select ix, stVector FROM PredicateFeatures WHERE ix = ?", (pfi, ))
    (pfi, stVector) = db.fetchone()
    vectorIndices = AddToAll(strToList(stVector), -1)
    return targetIndices, vectorIndices, ref_si, ref_pi



def openDB(fn):
    print 'opening ', fn
    conn = sqlite3.connect(fn)
    conn.text_factory = str   # this is new...
    return conn




def initializeAMRVectorDatabase(fn):
    db = openDB(fn)
    c = db.cursor()
    # Modified tables so that file size is smaller, moves more work into Lua, but recipe is programmable

    c.execute( 'DROP TABLE IF EXISTS       GeneralArch' )
    c.execute( 'CREATE TABLE IF NOT EXISTS GeneralArch( ix int, key text, value text, PRIMARY KEY (ix))' )

    c.execute( 'DROP TABLE IF EXISTS       DBFeatures' )
    c.execute( 'CREATE TABLE IF NOT EXISTS DBFeatures( type text, ix int, CSV text, PRIMARY KEY (ix, type))' )

    c.execute( 'DROP TABLE IF EXISTS       Tokens' )
    c.execute( 'CREATE TABLE IF NOT EXISTS Tokens( type text, ix int, token text, PRIMARY KEY (type, ix) )' )

    c.execute( 'DROP TABLE IF EXISTS       Items' )
    c.execute( 'CREATE TABLE IF NOT EXISTS Items( ix int, type text, sentIX int, sentenceLen int, pWordIX int, targetCSV text,' + \
               ' PRIMARY KEY (ix, type))' )

    c.execute( 'DROP TABLE IF EXISTS       Sentences' )
    c.execute( 'CREATE TABLE IF NOT EXISTS Sentences( ix int, type text, fType text, fCSV text, PRIMARY KEY (ix, type, fType))' )

    c.execute( 'DROP TABLE IF EXISTS       WDFArch' )
    c.execute( 'CREATE TABLE IF NOT EXISTS WDFArch( ix int, filename text, size int, width int, learningRate float, name text, clone text, ' + \
               ' ptrName text, fType text, offset int, PRIMARY KEY (ix))' )
    db.commit()
    return db


def summarizeDataFrame(inDF, groupCol, excludeVal=None, displayCols=[], countCol=None):    
    if 'object' != str(inDF[groupCol].dtype):
        z = inDF[0:0] # empty, but with same column structure
    else:
        z = inDF[inDF[groupCol] != excludeVal].copy()
    if not countCol:
        countCol='count'
        z[countCol]=1
    z = z.groupby(groupCol).count().sort_values(by=['count'], ascending=[0])
    z['cum_sum'] = z[countCol].cumsum()
    z['cum_perc'] = 100*z.cum_sum/z[countCol].sum()
    z['perc'] = 100*z[countCol]/z[countCol].sum()
    z['rank'] = range(len(z.index))
    if displayCols:    
        displayCols = [countCol, 'cum_sum', 'cum_perc', 'rank'] + displayCols
        return z[ displayCols ]
    else:
        return z[ [countCol,  'perc', 'cum_sum', 'cum_perc'] ]
    
def summarizeDataFrameMultiCols(inDF, groupCols, excludeVal=None, displayCols=[], countCol=None):
    zlist=[]
    for col in groupCols:    
        zlist += inDF[inDF[col].astype(str) != excludeVal][col].dropna().tolist()
    z = pd.DataFrame()    
    z['labels'] = pd.Series(zlist, name='labels')
    if not countCol:
        countCol='count'
        z[countCol]=1
    z = z.groupby('labels').count().sort_values(by=['count'], ascending=[0])
    z['cum_sum'] = z[countCol].cumsum()
    z['cum_perc'] = 100*z.cum_sum/z[countCol].sum()
    z['rank'] = range(len(z.index))
    return z
 
def summarizeDataFrameMultiColPairs(inDF, firstCols, secondCols ):
    list1=[]
    list2=[]
    for col in firstCols:    
        list1 += inDF[col].dropna().tolist()
    for col in secondCols:    
        list2 += inDF[col].dropna().tolist()
    joinedList=[]
    for i,t in enumerate(list1):
        joinedList.append(t + '(' + list2[i])    
    z = pd.DataFrame()    
    z['labels'] = pd.Series(joinedList, name='labels')
    countCol='count'
    z[countCol]=1
    z = z.groupby('labels').count().sort_values(by=['count'], ascending=[0])
    z['cum_sum'] = z[countCol].cumsum()
    z['cum_perc'] = 100*z.cum_sum/z[countCol].sum()
    z['rank'] = range(len(z.index))
    return z

def capsTag(w):
    capbool = [1 if c.isupper() else 0 for c in w]
    capsum = sum(capbool)
    if (capsum==0):
        caps = 'nocaps'
    elif ((capsum == 1) and (capbool[0]==1)):
        caps = 'initcap'
    elif (capsum == len(w)):
        caps = 'allcaps'
    else:
        caps = 'hascap'
    return caps

def getIndex(w, feature, defaultToken='UNKNOWN'):
    if (w == '') or pd.isnull(w):
        w = 'O'
    if w not in feature['t2i']:
        return feature['t2i'][defaultToken]
    else:
        return feature['t2i'][w]
    
    
    
    
def getdistSGFeatureInfo(vectors):
    f = { 'tokens' : [], 't2i' : {} }
    flist = []
    for sType in vectors:
        if not 'distSG' in vectors[sType]:
            return f
        flist += vectors[sType]['distSG'].tolist()
    flist.append(flist[-1])                               # <--------------------- This should be vectors for UNK and PAD
    flist.append(flist[-1])                               # <--------------------- This should be vectors for UNK and PAD
    #flist += ['UNKNOWN', 'PADDING']                      # <--------------------- This should be vectors for UNK and PAD
    f['tokens'] = flist
    f['t2i']    = dict( [(x, y) for y, x in enumerate(f['tokens'])] )
        
    return f

def getWordsFeatureInfo(dtVectors, randomWords, randomWordsCutoffPercent):
    if not randomWords:
        path = WordRepsFileLocations.pathToWordList()
        f = getFDef(path)
    else:
        sdf = summarizeDataFrame(dtVectors, 'words', excludeVal='O', displayCols=[ 'relSrc', 'ar0_arg', 'ar1_arg', 'ar2_arg' ])
        f={}
        f['tokens'] = sdf[sdf['cum_perc']<=randomWordsCutoffPercent].index.tolist() + ['UNKNOWN', 'PADDING']
        f['t2i']    = dict( [(x, y) for y, x in enumerate(f['tokens'])] )
    return f

 
def getCapsFeatureInfo():
    f={}
    f['tokens'] = ['PADDING', 'allcaps', 'hascap', 'initcap', 'nocaps']
    f['t2i']    = dict( [(x, y) for y, x in enumerate(f['tokens'])] )
    return f

def getDistanceFeatureInfo(maxD):
    f={}
    f['tokens']=[]
    for i in range(-maxD,maxD+1):
        f['tokens'].append('%d' % i)
    f['t2i']    = dict( [(x, y) for y, x in enumerate(f['tokens'])] )
    return f

def getSuffixFeatureInfo(dtVectors, suffixCutoffPercent):
    fullWordList = dtVectors['words'].tolist()
    suffices = [w.lower()[-2:] for w in fullWordList]
    dtVectors['suffix'] = suffices
    suffixSummary = summarizeDataFrame(dtVectors, 'suffix')
    f={}
    f['tokens']  = suffixSummary[suffixSummary['cum_perc']<=suffixCutoffPercent].index.tolist() + ['UNKNOWN', 'PADDING']
    f['t2i']    = dict( [(x, y) for y, x in enumerate(f['tokens'])] )
    return f
   
def getConceptFeatureInfo(dtVectors, cutoffPercent):
    sdf = summarizeDataFrame(dtVectors, 'txBIOES', excludeVal='O')
    f={}
    f['tokens']  = ['O'] + sdf[sdf['cum_perc']<=cutoffPercent].index.tolist() + ['UNKNOWN', 'PADDING']
    f['t2i']    = dict( [(x, y) for y, x in enumerate(f['tokens'])] )
    return f
   
def getNERFeatureInfo(dtVectors):
    sdf = summarizeDataFrame(dtVectors, 'NERLabel', excludeVal='O' )
    f={}
    f['tokens']  = ['O'] + sdf.index.tolist() + ['UNKNOWN', 'PADDING']
    f['t2i']    = dict( [(x, y) for y, x in enumerate(f['tokens'])] )
    return f
   
def getArgsFeatureInfo(dtVectors, cutoffPercent):
    argList = [ 'ar0_arg', 'ar1_arg', 'ar2_arg', 'ar3_arg' ]

    sdf = summarizeDataFrameMultiCols(dtVectors, argList,
                                       excludeVal='O' )
    f={}
    f['tokens']  = ['O'] + sdf[sdf['cum_perc']<=cutoffPercent].index.tolist() + ['UNKNOWN']
    f['t2i']    = dict( [(x, y) for y, x in enumerate(f['tokens'])] )
    return f
   
def getNargsFeatureInfo(dtVectors, cutoffPercent):
    nargList = [ 'nar0_lbl', 'nar1_lbl', 'nar2_lbl', 'nar3_lbl'  ]
    sdf = summarizeDataFrameMultiCols(dtVectors, nargList,
                                       excludeVal='O' )
    
    print sdf
    f={}
    f['tokens']  = ['O'] + sdf[sdf['cum_perc']<=cutoffPercent].index.tolist() + ['UNKNOWN']
    f['t2i']    = dict( [(x, y) for y, x in enumerate(f['tokens'])] )
    return f
   
def getAttrsFeatureInfo(dtVectors, cutoffPercent):
    attrList1 = [ 'attr0_lbl', 'attr1_lbl', 'attr2_lbl', 'attr3_lbl' ]
    attrList2 = [ 'attr0_val', 'attr1_val', 'attr2_val', 'attr3_val' ]
    sdf = summarizeDataFrameMultiColPairs(dtVectors, attrList1, attrList2 )
    print sdf
    f={}
    f['tokens']  = ['O'] + sdf[sdf['cum_perc']<=cutoffPercent].index.tolist() + ['UNKNOWN']
    f['t2i']    = dict( [(x, y) for y, x in enumerate(f['tokens'])] )
    return f

def getUnqualifiedAttrsFeatureInfo(dtVectors, cutoffPercent):

    sdf = summarizeDataFrameMultiCols(dtVectors, [ 'attr0_lbl', 'attr1_lbl', 'attr2_lbl', 'attr3_lbl' ] )
    print sdf
    f={}
    f['tokens']  = ['O'] + sdf[sdf['cum_perc']<=cutoffPercent].index.tolist() + ['UNKNOWN']
    f['t2i']    = dict( [(x, y) for y, x in enumerate(f['tokens'])] )
    return f

def getNCATFeatureInfo(dtVectors):
    sdf = summarizeDataFrame(dtVectors, 'nameCategory', excludeVal='O', displayCols=[ 'WCAT0', 'WCAT1', 'WCAT2', 'words' ])
    f={}
    f['tokens'] = ['O'] + sdf[sdf['cum_perc']<=97.0].index.tolist() + ['UNKNOWN', 'PADDING']
    f['t2i']    = dict( [(x, y) for y, x in enumerate(f['tokens'])] )
    return f

def getWCATFeatureInfo(dtVectors):
    stack = pd.DataFrame()
    for i in range(8):
        w = dtVectors[ ['WCAT%d'%i ] ]   
        w.columns = ['WCAT']
        stack = stack.append(w)
    sdf = summarizeDataFrame(stack, 'WCAT', excludeVal='O', displayCols=[ ])
    f={}
    f['tokens'] = ['O'] + sdf[sdf['cum_perc']<=76.0].index.tolist() + ['UNKNOWN', 'PADDING']
    f['t2i']    = dict( [(x, y) for y, x in enumerate(f['tokens'])] )
    return f

def checkFeatures(dtVectors, features):
    
    for key in features.keys():
        print key, len(features[key]['tokens'])
    sdf = summarizeDataFrame(dtVectors, 'words', excludeVal='O', displayCols=[ 'relSrc', 'ar0_arg', 'ar1_arg', 'ar2_arg' ])
    wordList = sdf.index.tolist()
    translatedWords = [parseWord(w, features['words']['t2i']) for w in wordList]
    backToWords     = [features['words']['tokens'][ix] for ix in translatedWords]
    sdf['translatedWords'] = translatedWords
    sdf['backToWords']     = backToWords
    unknownWords = sdf[sdf['backToWords']=='UNKNOWN']
    print unknownWords.head(100)

def createAMRL0Vectors(inFn, dbFn, L0CutoffPercent, keepSense, sTypes= ['test','training','dev'], vectors=None, featuresDB=None, maxSents=None, useNER=True):  
    wordDF={}
    if inFn:
        vectors = pickle.load(  open( inFn ) ) 
    vectors = preProcessVectors(vectors, sTypes, keepSense)
    for sType in sTypes:
        vectors[sType][ vectors[sType]['NERLabel']=='']['NERLabel'] = 'O'   # lazy way to correct initialization to '', can remove
    
    db = initializeAMRVectorDatabase(dbFn)

    # ================================

    # read features from the training database, or generate them?
    if featuresDB:
        _, features, _ = readAMRVectorDatabase(featuresDB)
    else:
        features = getAMRFeatures(vectors, L0CutoffPercent)    
                
    print 'add feature lists to db'
    featureNames = ['suffix', 'caps', 'words', 'L0']
    if useNER:
        featureNames += ['ner']
    for f in featureNames:
        for i,t in enumerate(features[f]['tokens']):
            db.execute("INSERT INTO Tokens (type, ix, token) VALUES (?, ?, ?)", (f, i+1, t))
                 
    db.execute("INSERT INTO GeneralArch (ix, key, value) VALUES (?, ?, ?)", ( 1, 'network', 'BDLSTM' ))
    db.execute("INSERT INTO GeneralArch (ix, key, value) VALUES (?, ?, ?)", ( 2, 'output', 'viterbi' ))
    db.execute("INSERT INTO GeneralArch (ix, key, value) VALUES (?, ?, ?)", ( 3, 'target', 'L0' ))

    cmd = "INSERT INTO WDFArch (ix, name, filename, size, width, learningRate, clone, ptrName, fType, offset) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    db.execute(cmd, ( 1, 'words',  WordRepsFileLocations.pathToWordRepsList(),  len(features['words']['tokens']), WordRepsFileLocations.wordRepsWidth(),  2.0,  None, 'wordP', 'words', 0 ) )
    db.execute(cmd, ( 2,  'caps',   None,                           len(features['caps']['tokens']),     5,  0.6,  None, 'wordP', 'caps', 0 ) )
    db.execute(cmd, ( 3,  'suffix', None,                           len(features['suffix']['tokens']),   5,  0.6,  None, 'wordP', 'suffix', 0 ) )
    if (useNER):
        db.execute(cmd, ( 4,  'ner',    None,                           len(features['ner']['tokens']),      5,  0.6,  None, 'wordP', 'ner', 0 ) )
        
    for sType in sTypes:   
        df = vectors[sType].dropna(subset = ['words']).copy()
        fullWordList = df['words'].tolist()
        BIOESList    = df['txBIOES'].tolist()
        
        df['suffIX'] = [getIndex(w.lower()[-2:], features['suffix'], defaultToken='UNKNOWN') for w in fullWordList]
        df['capsIX'] = [getIndex(capsTag(w),     features['caps'],   defaultToken='UNKNOWN') for w in fullWordList]
        df['wRepIX'] = [parseWord(w, features['words']['t2i']) for w in fullWordList]
        df['L0IX']   = [getIndex(b,              features['L0'],     defaultToken='UNKNOWN'      ) for b in BIOESList]
        if useNER:
            df['NERIX']  = [getIndex(n,              features['ner'],     defaultToken='UNKNOWN'      ) for n in df['NERLabel'].tolist()]
 
        # merge wRepIX into vectors, for DAMR construction
        wordDF[sType] = df[['sentIX','wordIX', 'words', 'wRepIX']] 

        maxIX = int(df[['sentIX']].values.max())
        if maxSents:
            maxIX= maxSents
        for sentIX in range(maxIX+1):
                  
            if not sentIX % 100:
                print 'creating vectors ', sType, sentIX, maxIX
            z = df[ df['sentIX'] == sentIX]
            if z.shape[0]>0:

                #print 'DEBUG ', sentIX, 'Z IS'
                #print z
                #print

                #len(z['words'].tolist()) 
                sentLength   = len(z['words'].tolist())
                tokensCSV    = ','.join(z['words'].tolist())
                suffixCSV    = listToCSV(AddToAll(z['suffIX'].tolist(), 1))
                capsCSV      = listToCSV(AddToAll(z['capsIX'].tolist(), 1)) 
                wordsCSV     = listToCSV(AddToAll(z['wRepIX'].tolist(), 1))  
                L0CSV        = listToCSV(AddToAll(z['L0IX'].tolist(), 1))  
                if useNER:
                    nerCSV       = listToCSV(AddToAll(z['NERIX'].tolist(), 1))  
                
                cmd = "INSERT INTO Sentences (ix, type, fType, fCSV) VALUES (?, ?, ?, ?)"
                db.execute(cmd, ( sentIX+1, sType, "words",  wordsCSV  ))
                db.execute(cmd, ( sentIX+1, sType, "caps",   capsCSV   ))
                db.execute(cmd, ( sentIX+1, sType, "suffix", suffixCSV ))
                db.execute(cmd, ( sentIX+1, sType, "tokens", tokensCSV ))
                db.execute(cmd, ( sentIX+1, sType, "L0",     L0CSV     ))
                if (useNER):
                    db.execute(cmd, ( sentIX+1, sType, "ner",    nerCSV     ))
                
                predIX = sentIX
                db.execute("INSERT INTO Items (ix, type, sentIX, sentenceLen, targetCSV) VALUES (?, ?, ?, ?, ?)", 
                               ( predIX+1, sType, sentIX+1, sentLength, L0CSV ))
                db.commit()       
    db.close()
    return wordDF


def preProcessVectors(vectors, sTypes, keepSense):        
    if (keepSense):
        for sType in sTypes:  
            temp  = vectors[sType]['txBIOES'].tolist() 
            sense = vectors[sType]['sense'].tolist() 
            print len(temp), len(sense)
            for i, t in enumerate(temp):
                if t=='S_txPred':
                    if (sense[i] in ['01','02']):   # NEW, only distinguish between 01 and 02.
                        t = t + '-' + sense[i]                             
                temp[i]=t    
            vectors[sType]['txBIOES'] = pd.Series(temp)
    # split WKCategory by /t to create the WCAT0-3
    for sType in sTypes:  
        temp  = vectors[sType]['WKCategory'].tolist() 
        wcat = []
        for i in range(8):
            wcat.append( [np.NaN] * len(temp) )
        for ti, toks in enumerate(temp):
            if not pd.isnull(toks):
                for i,t in enumerate(toks.split('\t')):
                    if len(t) and (i<len(wcat)):
                        wcat[i][ti] = t                             
        for i in range(len(wcat)):
            vectors[sType]['WCAT%d'%i] = pd.Series(wcat[i])
    return vectors

def getAMRFeatures(vectors, L0CutoffPercent):             
    randomWords              = False
    randomWordsCutoffPercent = 99.5
    suffixCutoffPercent      = 95
    print 'get Feature Lists'
    if ('dev' in vectors) and ('training' in vectors):
        devTestVectors = vectors['training'].append(vectors['dev'])
    else:
        devTestVectors = vectors['test']

    devTestL0Vectors = devTestVectors.copy().dropna(subset = ['words'])   # Level 0 only 

    pd.set_option('display.width',   10000)
    pd.set_option('display.max_rows', 200)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_colwidth', 200)
    
    features = {}
    features['distSG']   = getdistSGFeatureInfo(vectors)
    features['words']    = getWordsFeatureInfo(devTestL0Vectors, randomWords, randomWordsCutoffPercent)
    features['suffix']   = getSuffixFeatureInfo(devTestL0Vectors, suffixCutoffPercent)
    features['caps']     = getCapsFeatureInfo()
    features['distance'] = getDistanceFeatureInfo(10)
    features['L0']       = getConceptFeatureInfo(devTestL0Vectors, L0CutoffPercent)
    features['ner']      = getNERFeatureInfo(devTestL0Vectors)
    features['args']     = getArgsFeatureInfo(devTestL0Vectors, 100.0)
    features['nargs']    = getNargsFeatureInfo(devTestL0Vectors, 99.0)
    features['attr']     = getAttrsFeatureInfo(devTestL0Vectors, 100.0)
    features['ncat']     = getNCATFeatureInfo(devTestL0Vectors)
    features['wcat']     = getWCATFeatureInfo(devTestL0Vectors)
        
    return features
    
    
def getForcedProb(width, forcedIndex):
    maxP = 0.99
    p       =  [(1.0-maxP)/(width-1)] * width
    p[forcedIndex] = maxP
    return p

def createAMRL0NcatVectors(inFn, dbFn, L0CutoffPercent, keepSense, sTypes= ['test','training','dev'], 
                           vectors=None, featuresDB=None, maxSents=None, padVectorSG=None, 
                           L0OnlyFromFeaturesDB=False, useDistSG=False):  
    

    
    if inFn:
        vectors = pickle.load(  open( inFn ) ) 
    vectors = preProcessVectors(vectors, sTypes, keepSense)
    
    # There are three use cases:
    #
    # 0) create all features based on statistics of the data
    # 1) read all features from a database ( for running data forward through a generated model that does not use distSG )
    # 1s) read all features from a database, but use the DistSG as input ( for running data forward through a generated model that does not use distSG )
    # 2) useDistSG=True, read only L0 feature from a database, generate the others.  
    #       use the distSG column as input.
    
    if L0OnlyFromFeaturesDB:  # Used to form hard decisions from SG during training   
        _, vectorFeatures,  _ = readAMRVectorDatabase(featuresDB)
        features = getAMRFeatures(vectors, L0CutoffPercent)    
        features['L0']   = vectorFeatures['L0']  
    elif featuresDB: 
        _, features,   _ = readAMRVectorDatabase(featuresDB)
    else:
        features = getAMRFeatures(vectors, L0CutoffPercent)    
 
    db = initializeAMRVectorDatabase(dbFn)
     
    if useDistSG:
        features['distSG']   = getAMRFeatures(vectors, L0CutoffPercent)['distSG']  
        if len(features['distSG']['tokens'][0]) > 1:
            print 'SG feature is distributed and is %d by %d wide' %  (len(features['distSG']['tokens']), features['distSG']['tokens'][0].count(',')+ 1)
            # set the last vector to be the Padding vector, torch is coded to use this
            if not padVectorSG:
                SGWidth   = len( features['distSG']['tokens'][0].split(',') )
                padVectorSG = listToCSV(getForcedProb(SGWidth, 0))
            features['distSG']['tokens'].append(padVectorSG)
            features['distSG']['t2i']['PADDING'] = len(features['distSG']['tokens'])-1
            #features['distSG']['t2i']['UNKNOWN'] = len(features['distSG']['tokens'])-1  # fix added 6/4/17 during MNLI processing...
            
            SGFeature = 'distSG'
            SGWidth   = len( features['distSG']['tokens'][0].split(',') )
            SGColumn  = 'distSG'
            SGSource  = 'DB:distSG'  # this tells daisyluTorch to preload table from this DB
            print 'storing distSG feature to DB'
            for i,t in enumerate(features['distSG']['tokens']):
                db.execute("INSERT INTO DBFeatures (type, ix, CSV) VALUES (?, ?, ?)", ('distSG', i+1, t))
            SGLrate   = 0.0
        else:
            print 'Using hard decision from SG'
            for ss in ['training', 'dev', 'test']:
                iList = vectors[ss]['distSG'].tolist()
                rList = [features['L0']['tokens'][int(x)-1] for x in iList]  
                vectors[ss]['distSG_Prob'] = rList
            SGFeature ='L0'
            SGWidth   = 10
            SGColumn  = 'distSG_Prob'
            SGSource  = None
            SGLrate   = 1.0
        
    else:
        SGFeature ='L0'
        SGWidth   = 10
        SGColumn  = 'txBIOES'
        SGSource  = None
        SGLrate   = 1.0
    
     
    print(features.keys())
    print(features['ncat']['tokens'])
    print(features['wcat']['tokens'])
     
    print 'add feature lists to NCat db'
    for f in ['suffix', 'caps', 'words', 'L0', 'wcat', 'ncat']:
        print 'storing %s feature to DB' % f        
        for i,t in enumerate(features[f]['tokens']):
            db.execute("INSERT INTO Tokens (type, ix, token) VALUES (?, ?, ?)", (f, i+1, t))
                 
    db.execute("INSERT INTO GeneralArch (ix, key, value) VALUES (?, ?, ?)", ( 1, 'network', 'BDLSTM' ))
    db.execute("INSERT INTO GeneralArch (ix, key, value) VALUES (?, ?, ?)", ( 2, 'output',  'viterbi' ))
    db.execute("INSERT INTO GeneralArch (ix, key, value) VALUES (?, ?, ?)", ( 3, 'target',  'ncat' ))

    cmd = "INSERT INTO WDFArch (ix, name, filename, size, width, learningRate, clone, ptrName, fType, offset) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    db.execute(cmd, ( 1, 'words',   WordRepsFileLocations.pathToWordRepsList(),    len(features['words']['tokens']),    WordRepsFileLocations.wordRepsWidth(),  2.0,     None, 'wordP', 'words', 0  ))
    db.execute(cmd, ( 2, 'caps',       None,                           len(features['caps']['tokens']),     5,  0.6,  None, 'wordP', 'caps',     0  ))
    db.execute(cmd, ( 3, 'suffix',     None,                           len(features['suffix']['tokens']),   5,  0.6,  None, 'wordP', 'suffix',   0  ))
    db.execute(cmd, ( 4,  SGFeature, SGSource,                         len(features[SGFeature]['tokens']),  SGWidth,  SGLrate,  None, 'wordP', SGFeature,       0  ))


    db.execute(cmd, ( 5,  'wcat',        None,    len(features['wcat']['tokens']),    5,  0.6,   None,  'wordP', 'wcat', 0   ))
    db.execute(cmd, ( 6,  'wcat1',       None,    len(features['wcat']['tokens']),    5,  0.6,  'wcat', 'wordP', 'wcat', 0   ))
    db.execute(cmd, ( 7,  'wcat2',       None,    len(features['wcat']['tokens']),    5,  0.6,  'wcat', 'wordP', 'wcat', 0   ))
    db.execute(cmd, ( 8,  'wcat3',       None,    len(features['wcat']['tokens']),    5,  0.6,  'wcat', 'wordP', 'wcat', 0   ))
    db.execute(cmd, ( 9,  'wcat4',       None,    len(features['wcat']['tokens']),    5,  0.6,  'wcat', 'wordP', 'wcat', 0   ))
    db.execute(cmd, ( 10, 'wcat5',       None,    len(features['wcat']['tokens']),    5,  0.6,  'wcat', 'wordP', 'wcat', 0   ))
    db.execute(cmd, ( 11, 'wcat6',       None,    len(features['wcat']['tokens']),    5,  0.6,  'wcat', 'wordP', 'wcat', 0   ))
    db.execute(cmd, ( 12, 'wcat7',       None,    len(features['wcat']['tokens']),    5,  0.6,  'wcat', 'wordP', 'wcat', 0   ))
           

    
    for sType in sTypes:   # add ood
        predIX=0
        df = vectors[sType].dropna(subset = ['words']).copy()

        fullWordList = df['words'].tolist()
        BIOESList    = df[SGColumn].tolist() # cascaded from L0 nn
        fullWcat0List = df['WCAT0'].tolist()
        fullWcat1List = df['WCAT1'].tolist()
        fullWcat2List = df['WCAT2'].tolist()
        fullWcat3List = df['WCAT3'].tolist()
        fullWcat4List = df['WCAT4'].tolist()
        fullWcat5List = df['WCAT5'].tolist()
        fullWcat6List = df['WCAT6'].tolist()
        fullWcat7List = df['WCAT7'].tolist()
        fullNcatList  = df['nameCategory'].tolist()
        #----------
        
        df['suffIX'] = [getIndex(w.lower()[-2:], features['suffix'], defaultToken='UNKNOWN') for w in fullWordList]
        df['capsIX'] = [getIndex(capsTag(w),     features['caps'],   defaultToken='UNKNOWN') for w in fullWordList]
        df['wordIX'] = [parseWord(w, features['words']['t2i']) for w in fullWordList]
        df['L0IX']   = [getIndex(b,              features[SGFeature],     defaultToken='UNKNOWN'      ) for b in BIOESList]
        df['ncat']  = [getIndex(a,              features['ncat'],   defaultToken='UNKNOWN')       for a in fullNcatList]
        df['wcat0'] = [getIndex(a,              features['wcat'],   defaultToken='UNKNOWN')       for a in fullWcat0List]
        df['wcat1'] = [getIndex(a,              features['wcat'],   defaultToken='UNKNOWN')       for a in fullWcat1List]
        df['wcat2'] = [getIndex(a,              features['wcat'],   defaultToken='UNKNOWN')       for a in fullWcat2List]
        df['wcat3'] = [getIndex(a,              features['wcat'],   defaultToken='UNKNOWN')       for a in fullWcat3List]
        df['wcat4'] = [getIndex(a,              features['wcat'],   defaultToken='UNKNOWN')       for a in fullWcat4List]
        df['wcat5'] = [getIndex(a,              features['wcat'],   defaultToken='UNKNOWN')       for a in fullWcat5List]
        df['wcat6'] = [getIndex(a,              features['wcat'],   defaultToken='UNKNOWN')       for a in fullWcat6List]
        df['wcat7'] = [getIndex(a,              features['wcat'],   defaultToken='UNKNOWN')       for a in fullWcat7List]
    
        maxIX = int(df[['sentIX']].values.max())
        if maxSents:
            maxIX= maxSents
        for sentIX in range(maxIX+1):
            z = df[ df['sentIX'] == sentIX]
            if z.shape[0]>0:

                sentLength   = len(z['words'].tolist())
                #BIOES        = z['txBIOES'].tolist()
                wordsCSV     = listToCSV(AddToAll(z['wordIX'].tolist(), 1))  
                L0CSV        = listToCSV(AddToAll(z['L0IX'].tolist(), 1))  
                suffixCSV    = listToCSV(AddToAll(z['suffIX'].tolist(), 1))
                capsCSV      = listToCSV(AddToAll(z['capsIX'].tolist(), 1)) 
                ncatCSV       = listToCSV(AddToAll(z['ncat'].tolist(), 1)) 
                wcat0CSV      = listToCSV(AddToAll(z['wcat0'].tolist(), 1)) 
                wcat1CSV      = listToCSV(AddToAll(z['wcat1'].tolist(), 1)) 
                wcat2CSV      = listToCSV(AddToAll(z['wcat2'].tolist(), 1)) 
                wcat3CSV      = listToCSV(AddToAll(z['wcat3'].tolist(), 1)) 
                wcat4CSV      = listToCSV(AddToAll(z['wcat4'].tolist(), 1)) 
                wcat5CSV      = listToCSV(AddToAll(z['wcat5'].tolist(), 1)) 
                wcat6CSV      = listToCSV(AddToAll(z['wcat6'].tolist(), 1)) 
                wcat7CSV      = listToCSV(AddToAll(z['wcat7'].tolist(), 1)) 

                cmd = "INSERT INTO Sentences (ix, type, fType, fCSV) VALUES (?, ?, ?, ?)"
                db.execute(cmd, ( sentIX+1, sType, "words",  wordsCSV  ))
                db.execute(cmd, ( sentIX+1, sType, SGFeature,     L0CSV     ))
                db.execute(cmd, ( sentIX+1, sType, "caps",   capsCSV   ))
                db.execute(cmd, ( sentIX+1, sType, "suffix", suffixCSV ))
                db.execute(cmd, ( sentIX+1, sType, "wcat",   wcat0CSV ))
                db.execute(cmd, ( sentIX+1, sType, "wcat1",  wcat1CSV ))
                db.execute(cmd, ( sentIX+1, sType, "wcat2",  wcat2CSV ))
                db.execute(cmd, ( sentIX+1, sType, "wcat3",  wcat3CSV ))
                db.execute(cmd, ( sentIX+1, sType, "wcat4",  wcat4CSV ))
                db.execute(cmd, ( sentIX+1, sType, "wcat5",  wcat5CSV ))
                db.execute(cmd, ( sentIX+1, sType, "wcat6",  wcat6CSV ))
                db.execute(cmd, ( sentIX+1, sType, "wcat7",  wcat7CSV ))

                db.execute("INSERT INTO Items (ix, type, sentIX, pWordIX, sentenceLen, targetCSV) VALUES (?, ?, ?, ?, ?, ?)", 
                               ( predIX+1, sType, sentIX+1, i+1, sentLength, ncatCSV ))
                db.commit() 
                predIX += 1      
    db.commit()       
    db.close()

    
def createAMRL0ArgVectors(inFn, dbFn, L0CutoffPercent, keepSense, sTypes= ['test','training','dev'], 
                           vectors=None, featuresDB=None, maxSents=None, padVectorSG=None, 
                           L0OnlyFromFeaturesDB=False, useDistSG=False):  
    if inFn:
        vectors = pickle.load(  open( inFn ) ) 
    vectors = preProcessVectors(vectors, sTypes, keepSense)
    
    # There are three use cases:
    #
    # 0) create all features based on statistics of the data
    # 1) read all features from a database ( for running data forward through a generated model )
    # 2) useDistSG=True, read only L0 feature from a database, generate the others.  
    #       use the distSG column as input.
    
    if L0OnlyFromFeaturesDB:  # Used to form hard decisions from SG during training   
        _, vectorFeatures,  _ = readAMRVectorDatabase(featuresDB)
        features = getAMRFeatures(vectors, L0CutoffPercent)    
        features['L0'] = vectorFeatures['L0']
    elif featuresDB: 
        _, features,   _ = readAMRVectorDatabase(featuresDB)
    else:
        features = getAMRFeatures(vectors, L0CutoffPercent)    
    
    db = initializeAMRVectorDatabase(dbFn)

    
    if useDistSG:
        features['distSG']   = getAMRFeatures(vectors, L0CutoffPercent)['distSG']  
        if len(features['distSG']['tokens'][0]) > 1:
            print 'SG feature is distributed and is %d by %d wide' %  (len(features['distSG']['tokens']), features['distSG']['tokens'][0].count(',')+ 1)
            # set the last vector to be the Padding vector, torch is coded to use this
            if not padVectorSG:
                SGWidth   = len( features['distSG']['tokens'][0].split(',') )
                padVectorSG = listToCSV(getForcedProb(SGWidth, 0))
            features['distSG']['tokens'].append(padVectorSG)
            features['distSG']['t2i']['PADDING'] = len(features['distSG']['tokens'])-1
            
            SGFeature = 'distSG'
            SGWidth   = len( features['distSG']['tokens'][0].split(',') )
            SGColumn  = 'distSG'
            SGSource  = 'DB:distSG'  # this tells daisyluTorch to preload table from this DB
            print 'storing distSG feature to DB'
            for i,t in enumerate(features['distSG']['tokens']):
                db.execute("INSERT INTO DBFeatures (type, ix, CSV) VALUES (?, ?, ?)", ('distSG', i+1, t))
            SGLrate   = 0.0
        else:
            print 'Using hard decision from SG'
            for ss in ['training', 'dev', 'test']:
                iList = vectors[ss]['distSG'].tolist()
                rList = [features['L0']['tokens'][int(x)-1] for x in iList]  
                vectors[ss]['distSG_Prob'] = rList
            SGFeature ='L0'
            SGWidth   = 10
            SGColumn  = 'distSG_Prob'
            SGSource  = None
            SGLrate   = 1.0
        
    else:
        SGFeature ='L0'
        SGWidth   = 10
        SGColumn  = 'txBIOES'
        SGSource  = None
        SGLrate   = 1.0
    
    print(features.keys())
    print(features['args']['tokens'])
         
    print 'add feature lists to Args db'
    for f in ['suffix', 'caps', 'words', 'L0', 'args', 'distance']:
        print 'storing %s feature to DB' % f
        for i,t in enumerate(features[f]['tokens']):
            db.execute("INSERT INTO Tokens (type, ix, token) VALUES (?, ?, ?)", (f, i+1, t))

                 
    db.execute("INSERT INTO GeneralArch (ix, key, value) VALUES (?, ?, ?)", ( 1, 'network', 'BDLSTM' ))
    db.execute("INSERT INTO GeneralArch (ix, key, value) VALUES (?, ?, ?)", ( 2, 'output',  'viterbi' ))
    db.execute("INSERT INTO GeneralArch (ix, key, value) VALUES (?, ?, ?)", ( 3, 'target',  'args' ))

    cmd = "INSERT INTO WDFArch (ix, name, filename, size, width, learningRate, clone, ptrName, fType, offset) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    db.execute(cmd, ( 1, 'words',   WordRepsFileLocations.pathToWordRepsList(),    len(features['words']['tokens']),    WordRepsFileLocations.wordRepsWidth(),  2.0,  None, 'wordP', 'words', 0  ))
    db.execute(cmd, ( 2, 'caps',       None,                           len(features['caps']['tokens']),     5,  0.6,  None, 'wordP', 'caps', 0 ) )
    db.execute(cmd, ( 3, 'suffix',     None,                           len(features['suffix']['tokens']),   5,  0.6,  None, 'wordP', 'suffix', 0 ) )
    db.execute(cmd, ( 4, 'Pred',       None,                           len(features['words']['tokens']),    WordRepsFileLocations.wordRepsWidth(),  2.0,  "words", 'itemP', 'words', 0  ))
    db.execute(cmd, ( 5, 'ctxP1',      None,                           len(features['words']['tokens']),    WordRepsFileLocations.wordRepsWidth(),  2.0,  "words", 'itemP', 'words', -2  ))
    db.execute(cmd, ( 6, 'ctxP2',      None,                           len(features['words']['tokens']),    WordRepsFileLocations.wordRepsWidth(),  2.0,  "words", 'itemP', 'words', -1  ))
    db.execute(cmd, ( 7, 'ctxP3',      None,                           len(features['words']['tokens']),    WordRepsFileLocations.wordRepsWidth(),  2.0,  "words", 'itemP', 'words', 0  ))
    db.execute(cmd, ( 8, 'ctxP4',      None,                           len(features['words']['tokens']),    WordRepsFileLocations.wordRepsWidth(),  2.0,  "words", 'itemP', 'words', 1  ))
    db.execute(cmd, ( 9, 'ctxP5',      None,                           len(features['words']['tokens']),    WordRepsFileLocations.wordRepsWidth(),  2.0,  "words", 'itemP', 'words', 2  ))
    db.execute(cmd, ( 10, SGFeature,   SGSource,    len(features[SGFeature]['tokens']),   SGWidth,  SGLrate,  None,      'wordP', SGFeature, 0   ))
    db.execute(cmd, ( 11, 'L0Pred',     None,   len(features[SGFeature]['tokens']),       SGWidth,  SGLrate,  SGFeature, 'itemP', SGFeature, 0   ))
    db.execute(cmd, ( 12, 'L0ctxP1',    None,   len(features[SGFeature]['tokens']),       SGWidth,  SGLrate,  SGFeature, 'itemP', SGFeature, -2   ))
    db.execute(cmd, ( 13, 'L0ctxP2',    None,   len(features[SGFeature]['tokens']),       SGWidth,  SGLrate,  SGFeature, 'itemP', SGFeature, -1   ))
    db.execute(cmd, ( 14, 'L0ctxP3',    None,   len(features[SGFeature]['tokens']),       SGWidth,  SGLrate,  SGFeature, 'itemP', SGFeature, 0   ))
    db.execute(cmd, ( 15, 'L0ctxP4',    None,   len(features[SGFeature]['tokens']),       SGWidth,  SGLrate,  SGFeature, 'itemP', SGFeature, 1   ))
    db.execute(cmd, ( 16, 'L0ctxP5',    None,   len(features[SGFeature]['tokens']),       SGWidth,  SGLrate,  SGFeature, 'itemP', SGFeature, 2   ))
    db.execute(cmd, ( 17, 'regionMark', None,   len(features['distance']['tokens']),                        5,  0.6,  None, 'deltaP', 'distance', 0   ))
           

    
    for sType in sTypes:   # add ood
        predIX=0
        df = vectors[sType].dropna(subset = ['words']).copy()
        fullWordList = df['words'].tolist()
        BIOESList    = df[SGColumn].tolist()
        fullArg0List = df['ar0_arg'].tolist()
        fullArg1List = df['ar1_arg'].tolist()
        fullArg2List = df['ar2_arg'].tolist()
        fullArg3List = df['ar3_arg'].tolist()
        
        df['suffIX'] = [getIndex(w.lower()[-2:], features['suffix'], defaultToken='UNKNOWN') for w in fullWordList]
        df['capsIX'] = [getIndex(capsTag(w),     features['caps'],   defaultToken='UNKNOWN') for w in fullWordList]
        df['wordIX'] = [parseWord(w, features['words']['t2i']) for w in fullWordList]
        df['L0IX']   = [getIndex(b,              features[SGFeature],defaultToken='UNKNOWN'      ) for b in BIOESList]
        df['arg0IX'] = [getIndex(a,              features['args'],   defaultToken='UNKNOWN')       for a in fullArg0List]
        df['arg1IX'] = [getIndex(a,              features['args'],   defaultToken='UNKNOWN')       for a in fullArg1List]
        df['arg2IX'] = [getIndex(a,              features['args'],   defaultToken='UNKNOWN')       for a in fullArg2List]
        df['arg3IX'] = [getIndex(a,              features['args'],   defaultToken='UNKNOWN')       for a in fullArg3List]
    
        maxIX = int(df[['sentIX']].values.max())
        if maxSents:
            maxIX= maxSents
        for sentIX in range(maxIX+1):
            z = df[ df['sentIX'] == sentIX]
            if z.shape[0]>0:

                sentLength   = len(z['words'].tolist())
                BIOES        = z['txBIOES'].tolist()

                wordsCSV     = listToCSV(AddToAll(z['wordIX'].tolist(), 1))  
                L0CSV        = listToCSV(AddToAll(z['L0IX'].tolist(), 1))  
                suffixCSV    = listToCSV(AddToAll(z['suffIX'].tolist(), 1))
                capsCSV      = listToCSV(AddToAll(z['capsIX'].tolist(), 1)) 

                cmd = "INSERT INTO Sentences (ix, type, fType, fCSV) VALUES (?, ?, ?, ?)"
                db.execute(cmd, ( sentIX+1, sType, "words",  wordsCSV  ))
                db.execute(cmd, ( sentIX+1, sType, SGFeature,     L0CSV     ))
                db.execute(cmd, ( sentIX+1, sType, "caps",   capsCSV   ))
                db.execute(cmd, ( sentIX+1, sType, "suffix", suffixCSV ))

                # what to set
                arg0List     = z['arg0IX'].tolist()  
                arg1List     =  z['arg1IX'].tolist()  
                arg2List     =  z['arg2IX'].tolist()  
                arg3List     =  z['arg3IX'].tolist() 
                                
                arg0Loc = z['ar0_ix'].tolist()
                arg1Loc = z['ar1_ix'].tolist()
                arg2Loc = z['ar2_ix'].tolist()
                arg3Loc = z['ar3_ix'].tolist()
                           
                                
                for i,_ in enumerate(arg0Loc):
                    if ('txNonPred' in BIOES[i]) or ('txNamed' in BIOES[i]) or  ('O' == BIOES[i]) :  # this is how we figure out if ARGS come out of this node
                        continue
                    # the targetList contains the args in their proper positions
                    # Need to check what concepts are being trained here - ;)  
                    #
                    targetList = [0] * sentLength        # where 0 is the assumed null tag....            
                    if not pd.isnull(arg0Loc[i]):
                        if int(arg0Loc[i]) < len(targetList):
                            targetList[ int(arg0Loc[i]) ] = arg0List[i] 
                    if not pd.isnull(arg1Loc[i]):
                        if int(arg1Loc[i]) < len(targetList):
                            targetList[ int(arg1Loc[i]) ] = arg1List[i] 
                    if not pd.isnull(arg2Loc[i]):
                        if int(arg2Loc[i]) < len(targetList):
                            targetList[ int(arg2Loc[i]) ] = arg2List[i] 
                    if not pd.isnull(arg3Loc[i]):
                        if int(arg3Loc[i]) < len(targetList):
                            targetList[ int(arg3Loc[i]) ] = arg3List[i] 
                    #if (targetNotNull):
                    targetCSV = listToCSV(AddToAll(targetList, 1))  
                    db.execute("INSERT INTO Items (ix, type, sentIX, pWordIX, sentenceLen, targetCSV) VALUES (?, ?, ?, ?, ?, ?)", 
                                   ( predIX+1, sType, sentIX+1, i+1, sentLength, targetCSV ))
                    db.commit() 
                    predIX += 1      
    db.commit()       
    db.close()
        
def createAMRL0NargVectors(inFn, dbFn, L0CutoffPercent, keepSense, sTypes= ['test','training','dev'], 
                           vectors=None, featuresDB=None, maxSents=None, padVectorSG=None,
                           L0OnlyFromFeaturesDB=False, useDistSG=False):  
    if inFn:
        vectors = pickle.load(  open( inFn ) ) 
    vectors = preProcessVectors(vectors, sTypes, keepSense)
    
    # There are three use cases:
    #
    # 0) create all features based on statistics of the data
    # 1) read all features from a database ( for running data forward through a generated model )
    # 2) useDistSG=True, read only L0 feature from a database, generate the others.  
    #       use the distSG column as input.
    
    if L0OnlyFromFeaturesDB:  # Used to form hard decisions from SG during training   
        _, vectorFeatures,  _ = readAMRVectorDatabase(featuresDB)
        features = getAMRFeatures(vectors, L0CutoffPercent)    
        features['L0'] = vectorFeatures['L0']
    elif featuresDB: 
        _, features,   _ = readAMRVectorDatabase(featuresDB)
    else:
        features = getAMRFeatures(vectors, L0CutoffPercent)    
        
    db = initializeAMRVectorDatabase(dbFn)
 
    
    if useDistSG:
        features['distSG']   = getAMRFeatures(vectors, L0CutoffPercent)['distSG']  
        if len(features['distSG']['tokens'][0]) > 1:
            print 'SG feature is distributed and is %d by %d wide' %  (len(features['distSG']['tokens']), features['distSG']['tokens'][0].count(',')+ 1)
            # set the last vector to be the Padding vector, torch is coded to use this
            if not padVectorSG:
                SGWidth   = len( features['distSG']['tokens'][0].split(',') )
                padVectorSG = listToCSV(getForcedProb(SGWidth, 0))
            features['distSG']['tokens'].append(padVectorSG)
            features['distSG']['t2i']['PADDING'] = len(features['distSG']['tokens'])-1
            
            SGFeature = 'distSG'
            SGWidth   = len( features['distSG']['tokens'][0].split(',') )
            SGColumn  = 'distSG'
            SGSource  = 'DB:distSG'  # this tells daisyluTorch to preload table from this DB
            print 'storing distSG feature to DB'
            for i,t in enumerate(features['distSG']['tokens']):
                db.execute("INSERT INTO DBFeatures (type, ix, CSV) VALUES (?, ?, ?)", ('distSG', i+1, t))
            SGLrate   = 0.0
        else:
            print 'Using hard decision from SG'
            for ss in ['training', 'dev', 'test']:
                iList = vectors[ss]['distSG'].tolist()
                rList = [features['L0']['tokens'][int(x)-1] for x in iList]  
                vectors[ss]['distSG_Prob'] = rList
            SGFeature ='L0'
            SGWidth   = 10
            SGColumn  = 'distSG_Prob'
            SGSource  = None
            SGLrate   = 1.0
        
    else:
        SGFeature ='L0'
        SGWidth   = 10
        SGColumn  = 'txBIOES'
        SGSource  = None
        SGLrate   = 1.0
    
    print(features.keys())
    print(features['nargs']['tokens'])
    
    print 'add feature lists to Nargs db'
    for f in ['suffix', 'caps', 'words', 'L0', 'nargs', 'distance']:
        print 'storing %s feature to DB' % f
        for i,t in enumerate(features[f]['tokens']):
            db.execute("INSERT INTO Tokens (type, ix, token) VALUES (?, ?, ?)", (f, i+1, t))
                 
    db.execute("INSERT INTO GeneralArch (ix, key, value) VALUES (?, ?, ?)", ( 1, 'network', 'BDLSTM' ))
    db.execute("INSERT INTO GeneralArch (ix, key, value) VALUES (?, ?, ?)", ( 2, 'output',  'viterbi' ))
    db.execute("INSERT INTO GeneralArch (ix, key, value) VALUES (?, ?, ?)", ( 3, 'target',  'nargs' ))

    cmd = "INSERT INTO WDFArch (ix, name, filename, size, width, learningRate, clone, ptrName, fType, offset) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    db.execute(cmd, ( 1, 'words',   WordRepsFileLocations.pathToWordRepsList(),    len(features['words']['tokens']),    WordRepsFileLocations.wordRepsWidth(),  2.0,     None, 'wordP', 'words', 0  ))
    db.execute(cmd, ( 2, 'caps',       None,                           len(features['caps']['tokens']),     5,  0.6,  None, 'wordP', 'caps', 0 ) )
    db.execute(cmd, ( 3, 'suffix',     None,                           len(features['suffix']['tokens']),   5,  0.6,  None, 'wordP', 'suffix', 0 ) )
    db.execute(cmd, ( 4, 'Pred',       None,                           len(features['words']['tokens']),    WordRepsFileLocations.wordRepsWidth(),  2.0,  "words", 'itemP', 'words', 0  ))
    db.execute(cmd, ( 5, 'ctxP1',      None,                           len(features['words']['tokens']),    WordRepsFileLocations.wordRepsWidth(),  2.0,  "words", 'itemP', 'words', -2  ))
    db.execute(cmd, ( 6, 'ctxP2',      None,                           len(features['words']['tokens']),    WordRepsFileLocations.wordRepsWidth(),  2.0,  "words", 'itemP', 'words', -1  ))
    db.execute(cmd, ( 7, 'ctxP3',      None,                           len(features['words']['tokens']),    WordRepsFileLocations.wordRepsWidth(),  2.0,  "words", 'itemP', 'words', 0  ))
    db.execute(cmd, ( 8, 'ctxP4',      None,                           len(features['words']['tokens']),    WordRepsFileLocations.wordRepsWidth(),  2.0,  "words", 'itemP', 'words', 1  ))
    db.execute(cmd, ( 9, 'ctxP5',      None,                           len(features['words']['tokens']),    WordRepsFileLocations.wordRepsWidth(),  2.0,  "words", 'itemP', 'words', 2  ))
    db.execute(cmd, ( 10, SGFeature, SGSource,  len(features[SGFeature]['tokens']),      SGWidth,  SGLrate,  None,      'wordP', SGFeature, 0   ))
    db.execute(cmd, ( 11, 'L0Pred',     None,   len(features[SGFeature]['tokens']),      SGWidth,  SGLrate,  SGFeature, 'itemP', SGFeature, 0   ))
    db.execute(cmd, ( 12, 'L0ctxP1',    None,   len(features[SGFeature]['tokens']),      SGWidth,  SGLrate,  SGFeature, 'itemP', SGFeature, -2   ))
    db.execute(cmd, ( 13, 'L0ctxP2',    None,   len(features[SGFeature]['tokens']),      SGWidth,  SGLrate,  SGFeature, 'itemP', SGFeature, -1   ))
    db.execute(cmd, ( 14, 'L0ctxP3',    None,   len(features[SGFeature]['tokens']),      SGWidth,  SGLrate,  SGFeature, 'itemP', SGFeature, 0   ))
    db.execute(cmd, ( 15, 'L0ctxP4',    None,   len(features[SGFeature]['tokens']),      SGWidth,  SGLrate,  SGFeature, 'itemP', SGFeature, 1   ))
    db.execute(cmd, ( 16, 'L0ctxP5',    None,   len(features[SGFeature]['tokens']),      SGWidth,  SGLrate,  SGFeature, 'itemP', SGFeature, 2   ))
    db.execute(cmd, ( 17, 'regionMark', None,   len(features['distance']['tokens']),                        5,  0.6,  None, 'deltaP', 'distance', 0   ))
           

    
    for sType in sTypes:   
        predIX=0
        df = vectors[sType].dropna(subset = ['words']).copy()

        fullWordList = df['words'].tolist()
        BIOESList    = df[SGColumn].tolist()
        fullNarg0List = df['nar0_lbl'].tolist()
        fullNarg1List = df['nar1_lbl'].tolist()
        fullNarg2List = df['nar2_lbl'].tolist()
        fullNarg3List = df['nar3_lbl'].tolist()
        
        df['suffIX'] = [getIndex(w.lower()[-2:], features['suffix'], defaultToken='UNKNOWN') for w in fullWordList]
        df['capsIX'] = [getIndex(capsTag(w),     features['caps'],   defaultToken='UNKNOWN') for w in fullWordList]
        df['wordIX'] = [parseWord(w, features['words']['t2i']) for w in fullWordList]
        df['L0IX']   = [getIndex(b,               features[SGFeature], defaultToken='UNKNOWN'      ) for b in BIOESList]
        df['narg0IX'] = [getIndex(a,              features['nargs'],   defaultToken='UNKNOWN')       for a in fullNarg0List]
        df['narg1IX'] = [getIndex(a,              features['nargs'],   defaultToken='UNKNOWN')       for a in fullNarg1List]
        df['narg2IX'] = [getIndex(a,              features['nargs'],   defaultToken='UNKNOWN')       for a in fullNarg2List]
        df['narg3IX'] = [getIndex(a,              features['nargs'],   defaultToken='UNKNOWN')       for a in fullNarg3List]
    
        maxIX = int(df[['sentIX']].values.max())
        if maxSents:
            maxIX= maxSents
        for sentIX in range(maxIX+1):
            z = df[ df['sentIX'] == sentIX]
            if z.shape[0]>0:

                sentLength   = len(z['words'].tolist())
                BIOES        = z['txBIOES'].tolist()
                wordsCSV     = listToCSV(AddToAll(z['wordIX'].tolist(), 1))  
                L0CSV        = listToCSV(AddToAll(z['L0IX'].tolist(), 1))  
                suffixCSV    = listToCSV(AddToAll(z['suffIX'].tolist(), 1))
                capsCSV      = listToCSV(AddToAll(z['capsIX'].tolist(), 1)) 

                cmd = "INSERT INTO Sentences (ix, type, fType, fCSV) VALUES (?, ?, ?, ?)"
                db.execute(cmd, ( sentIX+1, sType, "words",   wordsCSV  ))
                db.execute(cmd, ( sentIX+1, sType, SGFeature, L0CSV     ))
                db.execute(cmd, ( sentIX+1, sType, "caps",    capsCSV   ))
                db.execute(cmd, ( sentIX+1, sType, "suffix",  suffixCSV ))

                # what to set
                narg0List     =  z['narg0IX'].tolist()  
                narg1List     =  z['narg1IX'].tolist()  
                narg2List     =  z['narg2IX'].tolist()  
                narg3List     =  z['narg3IX'].tolist() 
                                
                narg0Loc = z['nar0_ix'].tolist()
                narg1Loc = z['nar1_ix'].tolist()
                narg2Loc = z['nar2_ix'].tolist()
                narg3Loc = z['nar3_ix'].tolist()
               
                for i,_ in enumerate(narg0Loc):
                    if  ('O' == BIOES[i]) or ('txNamed' in BIOES[i]):
                        continue
                    # the targetList contains the args in their proper positions
                    targetList = [0] * sentLength                    
                    if not pd.isnull(narg0Loc[i]):
                        if int(narg0Loc[i]) < len(targetList):
                            targetList[ int(narg0Loc[i]) ] = narg0List[i] 
                    if not pd.isnull(narg1Loc[i]):
                        if int(narg1Loc[i]) < len(targetList):
                            targetList[ int(narg1Loc[i]) ] = narg1List[i] 
                    if not pd.isnull(narg2Loc[i]):
                        if int(narg2Loc[i]) < len(targetList):
                            targetList[ int(narg2Loc[i]) ] = narg2List[i] 
                    if not pd.isnull(narg3Loc[i]):
                        if int(narg3Loc[i]) < len(targetList):
                            targetList[ int(narg3Loc[i]) ] = narg3List[i] 
                    targetCSV = listToCSV(AddToAll(targetList, 1))  
                    db.execute("INSERT INTO Items (ix, type, sentIX, pWordIX, sentenceLen, targetCSV) VALUES (?, ?, ?, ?, ?, ?)", 
                                   ( predIX+1, sType, sentIX+1, i+1, sentLength, targetCSV ))
                    db.commit() 
                    predIX += 1      
    db.commit()       
    db.close()

def createAMRL0AttrVectors(inFn, dbFn, L0CutoffPercent, keepSense, sTypes= ['test','training','dev'], 
                           vectors=None, featuresDB=None, maxSents=None, padVectorSG=None,
                           L0OnlyFromFeaturesDB=False, useDistSG=False):  
    if inFn:
        vectors = pickle.load(  open( inFn ) ) 
    vectors = preProcessVectors(vectors, sTypes, keepSense)
    
    # There are three use cases:
    #
    # 0) create all features based on statistics of the data
    # 1) read all features from a database ( for running data forward through a generated model )
    # 2) useDistSG=True, read only L0 feature from a database, generate the others.  
    #       use the distSG column as input.
    
    if L0OnlyFromFeaturesDB:  # Used to form hard decisions from SG during training   
        _, vectorFeatures,  _ = readAMRVectorDatabase(featuresDB)
        features = getAMRFeatures(vectors, L0CutoffPercent)    
        features['L0'] = vectorFeatures['L0']
    elif featuresDB: 
        _, features,   _ = readAMRVectorDatabase(featuresDB)
    else:
        features = getAMRFeatures(vectors, L0CutoffPercent)    
    
    db = initializeAMRVectorDatabase(dbFn)

    
    if useDistSG:
        features['distSG']   = getAMRFeatures(vectors, L0CutoffPercent)['distSG']  
        if len(features['distSG']['tokens'][0]) > 1:
            print 'SG feature is distributed and is %d by %d wide' %  (len(features['distSG']['tokens']), features['distSG']['tokens'][0].count(',')+ 1)
            # set the last vector to be the Padding vector, torch is coded to use this
            if not padVectorSG:
                SGWidth   = len( features['distSG']['tokens'][0].split(',') )
                padVectorSG = listToCSV(getForcedProb(SGWidth, 0))
            features['distSG']['tokens'].append(padVectorSG)
            features['distSG']['t2i']['PADDING'] = len(features['distSG']['tokens'])-1
            
            SGFeature = 'distSG'
            SGWidth   = len( features['distSG']['tokens'][0].split(',') )
            SGColumn  = 'distSG'
            SGSource  = 'DB:distSG'  # this tells daisyluTorch to preload table from this DB
            print 'storing distSG feature to DB'
            for i,t in enumerate(features['distSG']['tokens']):
                db.execute("INSERT INTO DBFeatures (type, ix, CSV) VALUES (?, ?, ?)", ('distSG', i+1, t))
            SGLrate   = 0.0
        else:
            print 'Using hard decision from SG'
            for ss in ['training', 'dev', 'test']:
                iList = vectors[ss]['distSG'].tolist()
                rList = [features['L0']['tokens'][int(x)-1] for x in iList]  
                vectors[ss]['distSG_Prob'] = rList
            SGFeature ='L0'
            SGWidth   = 10
            SGColumn  = 'distSG_Prob'
            SGSource  = None
            SGLrate   = 1.0
        
    else:
        SGFeature ='L0'
        SGWidth   = 10
        SGColumn  = 'txBIOES'
        SGSource  = None
        SGLrate   = 1.0

    """
    Do just the top unqualified labels for now
    polarity                    
    TOP                          
    quant                         
    
    These could be added later
    mode(interrogative           709     8796   13.060134      5
    mode(imperative              328    15719   23.339272     21
    """
    features['attr']['tokens']  = ['O'] + ['polarity','TOP','quant'] + ['UNKNOWN']
    features['attr']['t2i']     = dict( [(x, y) for y, x in enumerate(features['attr']['tokens'])] )
    
    print(features.keys())
    print(features['attr']['tokens'])
       
         
    print 'add feature lists to Attr db'
    for f in ['suffix', 'caps', 'words', 'L0', 'attr', 'distance']:
        print 'storing %s feature to DB' % f
        for i,t in enumerate(features[f]['tokens']):
            db.execute("INSERT INTO Tokens (type, ix, token) VALUES (?, ?, ?)", (f, i+1, t))
                 
    db.execute("INSERT INTO GeneralArch (ix, key, value) VALUES (?, ?, ?)", ( 1, 'network', 'BDLSTM' ))
    db.execute("INSERT INTO GeneralArch (ix, key, value) VALUES (?, ?, ?)", ( 2, 'output',  'viterbi' ))
    db.execute("INSERT INTO GeneralArch (ix, key, value) VALUES (?, ?, ?)", ( 3, 'target',  'attr' ))

    cmd = "INSERT INTO WDFArch (ix, name, filename, size, width, learningRate, clone, ptrName, fType, offset) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    db.execute(cmd, ( 1, 'words',   WordRepsFileLocations.pathToWordRepsList(),    len(features['words']['tokens']),    WordRepsFileLocations.wordRepsWidth(),  2.0,     None, 'wordP', 'words', 0  ))
    db.execute(cmd, ( 2, 'caps',       None,                           len(features['caps']['tokens']),     5,  0.6,  None, 'wordP', 'caps',     0  ))
    db.execute(cmd, ( 3, 'suffix',     None,                           len(features['suffix']['tokens']),   5,  0.6,  None, 'wordP', 'suffix',   0  ))
    db.execute(cmd, ( 4, SGFeature,    SGSource,                       len(features[SGFeature]['tokens']), SGWidth,  SGLrate,  None, 'wordP', SGFeature,       0  ))
           

    
    for sType in sTypes:   # add ood
        predIX=0
        df = vectors[sType].dropna(subset = ['words']).copy()

        fullWordList = df['words'].tolist()
        BIOESList    = df[SGColumn].tolist()
        fullAttr0List = df['attr0_lbl'].tolist()
        fullAttr1List = df['attr1_lbl'].tolist()
        fullAttr2List = df['attr2_lbl'].tolist()
        fullAttr3List = df['attr3_lbl'].tolist()
        #----------
        
        df['suffIX'] = [getIndex(w.lower()[-2:], features['suffix'], defaultToken='UNKNOWN') for w in fullWordList]
        df['capsIX'] = [getIndex(capsTag(w),     features['caps'],   defaultToken='UNKNOWN') for w in fullWordList]
        df['wordIX'] = [parseWord(w, features['words']['t2i']) for w in fullWordList]
        df['L0IX']   = [getIndex(b,              features[SGFeature],     defaultToken='UNKNOWN'      ) for b in BIOESList]
        df['attr0IX'] = [getIndex(a,              features['attr'],   defaultToken='UNKNOWN')       for a in fullAttr0List]
        df['attr1IX'] = [getIndex(a,              features['attr'],   defaultToken='UNKNOWN')       for a in fullAttr1List]
        df['attr2IX'] = [getIndex(a,              features['attr'],   defaultToken='UNKNOWN')       for a in fullAttr2List]
        df['attr3IX'] = [getIndex(a,              features['attr'],   defaultToken='UNKNOWN')       for a in fullAttr3List]
    
        maxIX = int(df[['sentIX']].values.max())
        if maxSents:
            maxIX= maxSents
        for sentIX in range(maxIX+1):
            z = df[ df['sentIX'] == sentIX]
            if z.shape[0]>0:

                sentLength   = len(z['words'].tolist())
                BIOES        = z['txBIOES'].tolist()
                wordsCSV     = listToCSV(AddToAll(z['wordIX'].tolist(), 1))  
                L0CSV        = listToCSV(AddToAll(z['L0IX'].tolist(), 1))  
                suffixCSV    = listToCSV(AddToAll(z['suffIX'].tolist(), 1))
                capsCSV      = listToCSV(AddToAll(z['capsIX'].tolist(), 1)) 

                cmd = "INSERT INTO Sentences (ix, type, fType, fCSV) VALUES (?, ?, ?, ?)"
                db.execute(cmd, ( sentIX+1, sType, "words",  wordsCSV  ))
                db.execute(cmd, ( sentIX+1, sType, SGFeature,     L0CSV     ))
                db.execute(cmd, ( sentIX+1, sType, "caps",   capsCSV   ))
                db.execute(cmd, ( sentIX+1, sType, "suffix", suffixCSV ))

                # what to set
                attrList=[ []  ] * 4
                attrList[0]     =  z['attr0IX'].tolist()  
                attrList[1]     =  z['attr1IX'].tolist()  
                attrList[2]     =  z['attr2IX'].tolist()  
                attrList[3]     =  z['attr3IX'].tolist()  
              
                targetList = [0] * sentLength                    
                for w in range(sentLength):
                    #if any of the attr lists contains a token in the features dict,
                    #set it as the target for this word.
                    if not ( pd.isnull(BIOES[w])):
                        if attrList[0][w]>0 :
                            targetList[ w ] = attrList[0][w] 
                targetCSV = listToCSV(AddToAll(targetList, 1))  
                db.execute("INSERT INTO Items (ix, type, sentIX, pWordIX, sentenceLen, targetCSV) VALUES (?, ?, ?, ?, ?, ?)", 
                               ( predIX+1, sType, sentIX+1, i+1, sentLength, targetCSV ))
                db.commit() 
                predIX += 1      
    db.commit()       
    db.close()


    
def readAMRVectorDatabase(dbFn):
    db = openDB(dbFn)
    cur = db.cursor()
    sType = 'test'
               
    cmd = "SELECT * FROM Sentences WHERE (type = '%s')" % (sType)
    #print dbFn
    #print cmd
    cur.execute(cmd)
    keys = [z[0] for z in cur.description]
    sinfo = {}
    winfo = {}
    for row in cur:
        d = dict(zip(keys,row))
        if d['fType']=='tokens':
            sinfo[d['ix']] = d['fCSV']
        if d['fType']=='words':
            winfo[d['ix']] = d['fCSV']
    
    cmd = "SELECT * FROM Items WHERE (type = '%s')" % (sType)
    #print cmd
    cur.execute(cmd)
    keys = [z[0] for z in cur.description]
    rows = []
    for row in cur:
        d = dict(zip(keys,row))
        sentIX = d['sentIX']
        d['words'] = winfo[sentIX]
        if sentIX in sinfo:
            wordTokens = sinfo[sentIX]
            d['wordTokens'] = wordTokens

        rows.append(d)

        #print d['wordTokens']
        #srows.append(d)
        
    
        
        
    cmd = "SELECT * FROM Tokens"
    print cmd
    cur.execute(cmd)
    keys = [z[0] for z in cur.description]
    features = {}
    for row in cur:
        d = dict(zip(keys,row))
        tp = d['type']
        token = d['token']
        ix = d['ix'] - 1                               # adjusting for lua one based arrays
        if not tp in features:
            features[tp] = {'tokens':[], 't2i':{}}
        features[tp]['tokens'].append(token)
        features[tp]['t2i'][token]=ix
        if (ix != len(features[tp]['tokens'])-1):
            assert('Error in read tokens from db')
    cmd = "SELECT * FROM GeneralArch"
    print cmd
    cur.execute(cmd)
    keys = [z[0] for z in cur.description]
    generalArch = {}
    for row in cur:
        d = dict(zip(keys,row))
        key = d['key']
        val = d['value']
        generalArch[key] = val
        
    db.close()
    return pd.DataFrame(rows), features, generalArch 
    
def readAMRResultsDatabase(dbFn, sType = 'test'):
    db = openDB(dbFn)
    cur = db.cursor()

    """
  db:execute( 'CREATE TABLE IF NOT EXISTS Sentences( ix int, type text, targetVector text, PRIMARY KEY (ix, type))' )
    """
    
    cmd = "SELECT * FROM Items WHERE (type = '%s') ORDER BY ix;" % (sType)
    print cmd
    cur.execute(cmd)
    keys = [z[0] for z in cur.description]
    rows = []
    for row in cur:
        d = dict(zip(keys,row))
        rows.append(d)
        
        
    cmd = "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='%s';" % ('Parameters')
    cur.execute(cmd)
    if cur.fetchall()[0][0]:
        #type, dataString
        cmd = "SELECT * FROM Parameters WHERE (type = '%s')" % ('weightString')
        print cmd
        cur.execute(cmd)
        #rString = cur.fetchall()[0][1]
        #lst = floatCSVToList(rString) 
        #x = np.array(lst)
        x=None
        
        
    db.close()
    return pd.DataFrame(rows)


def getComparisonDFrames(dbfn, dbrfn, pVector2d=False):    
    # from vectors and results, compute a merged comparison [pandas dataframe]
    df, features, genArch = readAMRVectorDatabase(dbfn)
    targetTokenType = genArch['target']
    
    dfr = readAMRResultsDatabase(dbrfn)    
        
    result = pd.merge(df, dfr, on='ix')
    #
    # merge df with dfr based on 'ix'
    #   ix is the itemIX
    #   sentIX and pWordIX come from df
    #   make sure this still works on AMRL0, though
    #
    tokPairs = []
    confusion = {}
    for _,c_row in result.iterrows():
        sentIX  = c_row['sentIX']
        pWordIX = c_row['pWordIX']
        
        wstring = c_row['words']
        wi = wstring.split(',')
        wordTokens = [features['words']['tokens'][int(i)-1] for i in wi]
        tv = intCSVToList(c_row['targetCSV'])
        rv = intCSVToList(c_row['resultVector'])
        

        
        
        pVectors = c_row['logProbVector'].split('#')
        
        length = min(100,len(tv) )
        for i in range(length):
            ftv = features[targetTokenType]['tokens'][ tv[i]-1 ]
            frv = features[targetTokenType]['tokens'][ rv[i]-1 ]
            if pVector2d:
                if (pWordIX-1)==i:
                    pVector = c_row['logProbVector']
                else:    
                    pVector=None
            else:
                pVector = pVectors[i]
            tokPairs.append( {'sentIX':sentIX, 'pWordIX':pWordIX, 'wordIX':i, 'word':wordTokens[i],'target':ftv, 'result':frv, 'pVector':pVector} )  
            if not frv in confusion:
                confusion[frv] = {}
            if not ftv in confusion[frv]:
                confusion[frv][ftv] = 0    
            confusion[frv][ftv] += 1           
    tp = pd.DataFrame(tokPairs)
    return tp, df, dfr, features, genArch

def plotHeatmaps(tp, genArch=None):
    import seaborn as sns; sns.set()
    
    direc = getSystemPath('figures')
    prefix = 'heatmap'
    
    # tp contains target, result, count triplets
    # find the list of targets and the error counts associated with them
    # whats the accuracy per target?
    x = tp.groupby([ 'target', 'result'   ], as_index=False ).count()
    tList = tp[ tp['target'] != tp['result']].groupby( ['target'], as_index=False).count().sort(['sentIX'], ascending=[0])['target'].tolist()
    x = x[ x['target'].isin(tList[1:11])  ]
    #print x.sort(['target','sentIX'], ascending=[1,0])
    x = x.pivot("result", "target", "wordIX")

    title = 'Top 10, excluding O, most confused Target Tags'
    plt.figure(figsize=(18, 10))
    plt.yticks(rotation=0) 
    plt.xticks(rotation=90) 
    plt.title(title)
    plt.tight_layout()
    saveFn = '%s/%s_%d.png' % (direc, prefix, 1)
    plt.savefig(saveFn)

    
    # tp contains target, result, count triplets
    # find the list of targets and the error counts associated with them
    # whats the accuracy per target?
    x = tp.groupby([ 'target', 'result'   ], as_index=False ).count()
    tList = tp[ tp['target'] != tp['result']].groupby( ['target'], as_index=False).count().sort(['sentIX'], ascending=[0])['target'].tolist()
    x = x[ x['target'].isin(tList[:10])  ]
    #print x.sort(['target','sentIX'], ascending=[1,0])
    x = x.pivot("result", "target", "wordIX")

    title = 'Top 10 most confused Target Tags'
    plt.figure(figsize=(18, 10))
    plt.yticks(rotation=0) 
    plt.xticks(rotation=90) 
    plt.title(title)
    plt.tight_layout()
    saveFn = '%s/%s_%d.png' % (direc, prefix, 2)
    plt.savefig(saveFn)

    # tp contains target, result, count triplets
    # find the list of targets and the error counts associated with them
    # whats the accuracy per target?
    x = tp.groupby([ 'target', 'result'   ], as_index=False ).count()
    tList = tp[ tp['target'] != tp['result']].groupby( ['target'], as_index=False).count().sort(['sentIX'], ascending=[0])['target'].tolist()
    x = x.pivot("result", "target", "wordIX")
    
    title = 'Confusion Matrix for all Tags'
    plt.figure(figsize=(18, 10))
    plt.yticks(rotation=0) 
    plt.xticks(rotation=90) 
    plt.title(title)
    plt.tight_layout()
    saveFn = '%s/%s_%d.png' % (direc, prefix, 3)
    plt.savefig(saveFn)
    
    plt.show()
    sns.plt.show()

 
if __name__ == '__main__':
    
    exit(10)
    

  
    
    
    
