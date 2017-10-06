import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt
import collections
from pprint import pprint

from daisylu_vectors import *
# 
# merged here: from daisylu_new_dryrun import *
from daisylu_config import *
from daisylu_system import *
from sentences import *

import networkx as nx 
import argparse
import subprocess

import threading
import time
import sys


#
# 1 merge daisylu_new_dryrun to here
# 2 daisylu_dryrun?
#
#
   

def addTorchNetworkResults(sents, dbrf, dbf, systemName, NoConceptThreshold = 0.5, conceptRejectionThreshold=0.0):

    def logPMToProb(logPM):
        pMass       = [math.exp(float(z)) for z in logPM ]
        probs       = [pm/sum(pMass) for pm in pMass]
        return probs
    
    dbfn = getSystemPath('daisylu') + 'data/%s' % dbf
    dbrfn = getSystemPath('daisylu') + 'results/%s' % dbrf
    if (systemName == 'AMRL0'):
        # 2c. add L0 nn output to data frames
        tp, _, _, features, _ = getComparisonDFrames(dbfn, dbrfn)    
        for sentIX in range(len(sents['test'])):  
            sentence = sents['test'][sentIX]        
            singleSentDF = tp[ tp['sentIX']==(sentIX+1) ]        
            df = sentence.predictedDFrame 
            for _, row in singleSentDF.iterrows():
                wordIX  =  row['wordIX']
                result  =  row['result']
                pVector =  row['pVector']
                df.loc[df.wordIX == wordIX, 'pVectorL0'] = pVector
                lProb = np.array(floatCSVToList(pVector)[0]) 
               
                if (True):      
                    # With:                F1 57.18,  prec 59.71,  recall 54.85
                    # With 0.65 threshold: F1 57.49,  prec 58.09,  recall 56.90
                    # Without:             F1 57.35,  prec 60.57,  recall 54.46
                    feats = features['L0']['tokens']
                    lst = floatCSVToList(pVector) 
                    logProbs, probs = normalizeLogProbs1d({0:lst[0]})
                    L0ToProb = dict(zip(feats,probs[0]))
                    sortedTuples = sorted(zip(feats,probs[0]), key=lambda x: x[1], reverse=True )
                    if sortedTuples[0][0] == 'O':
                        if sortedTuples[0][1] >= NoConceptThreshold:
                            result = 'O'
                            prob = sortedTuples[0][1]
                        else:
                            result = sortedTuples[1][0]
                            prob = sortedTuples[1][1]
                    else:
                        result = sortedTuples[0][0]
                        prob = sortedTuples[0][1]
                    if result=='UNKNOWN':
                        if not 'NEWPrediction' in df:
                            result = 'S_txPred-01'  # a default       
                        else:
                            result = 'S_txPred'  # NEW default, was S_txPred-01       
                
                if (conceptRejectionThreshold > 0.0) and not (result=='txNamed'):
                    if prob <= conceptRejectionThreshold:
                        result='O'
                            
                dsfString = listToCSV(logPMToProb(lProb))
                df.loc[df.wordIX == wordIX, 'distSG']      = dsfString
                df.loc[df.wordIX == wordIX, 'txBIOES']     = result
                df.loc[df.wordIX == wordIX, 'txBIOESProb'] = prob
                        
            # 2d. add wikification to data frames (alternate is to run wikification first and use NER as input to L0
            # Wikification should happen here, creating the 'namedCategory'
            # and the 'wiki' attribute
            df['nameCategory'] = 'GetFromWikification'
            df['wiki']         = 'GetFromWikification'
            if not 'NEWPrediction' in df:
                predictConceptKinds(df, None)
    elif (systemName == 'AMRL0Args'):
        tp, df, _, _, _ = getComparisonDFrames(dbfn, dbrfn, pVector2d=True)   
        for sentIX in range(len(sents['test'])):  
            sentence = sents['test'][sentIX]        
            singleSentDF = tp[ tp['sentIX']==(sentIX+1) ]        
            df = sentence.predictedDFrame 
            argsDict = {}
            for _, row in singleSentDF.iterrows():
                pWordIX =  row['pWordIX']-1    # adjust for lua
                wordIX  =  row['wordIX']
                result  =  row['result']
                pVector =  row['pVector']
                if pVector:
                    df.loc[df.wordIX == pWordIX, 'pVectorL0Args'] = pVector
                
                if result != 'O':
                    if not pWordIX in argsDict:
                        argsDict[pWordIX] = []
                    argsDict[pWordIX].append([wordIX, result])
            for pWordIX in argsDict.keys():
                for i,rel in enumerate(argsDict[pWordIX]):
                    df.loc[df.wordIX == pWordIX, 'ar%d_ix' % i]  = rel[0]
                    df.loc[df.wordIX == pWordIX, 'ar%d_arg' % i] = rel[1]
    elif (systemName == 'AMRL0Nargs'):
        # 2c. add L0 nn output to data frames
        tp, df, _, _, _ = getComparisonDFrames(dbfn, dbrfn, pVector2d=True)   
        for sentIX in range(len(sents['test'])):  
            sentence = sents['test'][sentIX]        
            singleSentDF = tp[ tp['sentIX']==(sentIX+1) ]        
            df = sentence.predictedDFrame 
            nargsDict = {}
            for _, row in singleSentDF.iterrows():
                pWordIX =  row['pWordIX']-1    # adjust for lua
                wordIX  =  row['wordIX']    
                result  =  row['result']
                pVector =  row['pVector']
                if pVector:
                    df.loc[df.wordIX == wordIX, 'pVectorL0Nargs'] = pVector
                if result != 'O':
                    if not pWordIX in nargsDict:
                        nargsDict[pWordIX] = []
                    nargsDict[pWordIX].append([wordIX, result])
                #print 'DEBUG', pWordIX, wordIX, result
            for pWordIX in nargsDict.keys():
                for i,rel in enumerate(nargsDict[pWordIX]):
                    df.loc[df.wordIX == pWordIX, 'nar%d_ix' % i]  = rel[0]
                    df.loc[df.wordIX == pWordIX, 'nar%d_lbl' % i] = rel[1]
    elif (systemName == 'AMRL0Attr'):
        # 2c. add L0 nn output to data frames
        tp, df, _, _, _ = getComparisonDFrames(dbfn, dbrfn)   
        for sentIX in range(len(sents['test'])):  
            sentence = sents['test'][sentIX]        
            singleSentDF = tp[ tp['sentIX']==(sentIX+1) ]        
            df = sentence.predictedDFrame 
            for _, row in singleSentDF.iterrows():
                pWordIX =  row['pWordIX']-1    # adjust for lua
                wordIX  =  row['wordIX']
                result  =  row['result']
                pVector =  row['pVector']
                df.loc[df.wordIX == wordIX, 'pVectorL0Attr'] = pVector   
                i=0             
                if result == 'polarity':
                    df.loc[df.wordIX == wordIX, 'attr%d_val' % i] = '-'
                    df.loc[df.wordIX == wordIX, 'attr%d_lbl' % i] = 'polarity'
                elif result == 'TOP':
                    df.loc[df.wordIX == wordIX, 'attr%d_val' % i] = df.loc[df.wordIX == wordIX, 'kind'] 
                    df.loc[df.wordIX == wordIX, 'attr%d_lbl' % i] = 'TOP'
                elif result == 'quant':
                    print 'skipping quant HMM'
                    #df.loc[df.wordIX == wordIX, 'attr%d_val' % i] = 'HMM'
                    #df.loc[df.wordIX == wordIX, 'attr%d_lbl' % i] = 'quant'
                    

    if (systemName == 'AMRL0Ncat'):
        # 2c. add L0 nn output to data frames
        tp, _, _, features, _ = getComparisonDFrames(dbfn, dbrfn)    
        for sentIX in range(len(sents['test'])):  
            sentence = sents['test'][sentIX]        
            singleSentDF = tp[ tp['sentIX']==(sentIX+1) ]        
            df = sentence.predictedDFrame 
            for _, row in singleSentDF.iterrows():
                wordIX =  row['wordIX']
                result =  row['result']
                pVector =  row['pVector']
                df.loc[df.wordIX == wordIX, 'pVectorNcat'] = pVector
                
                if (True):      
                    # With:                F1 57.18,  prec 59.71,  recall 54.85
                    # With 0.65 threshold: F1 57.49,  prec 58.09,  recall 56.90
                    # Without:             F1 57.35,  prec 60.57,  recall 54.46
                    feats = features['ncat']['tokens']
                    lst = floatCSVToList(pVector) 
                    logProbs, probs = normalizeLogProbs1d({0:lst[0]})
                    L0ToProb = dict(zip(feats,probs[0]))
                    sortedTuples = sorted(zip(feats,probs[0]), key=lambda x: x[1], reverse=True )
                    if sortedTuples[0][0] == 'O':
                        if sortedTuples[0][1] >= 0.95:
                            result = '-'
                            prob = sortedTuples[0][1]
                        else:
                            result = sortedTuples[1][0]
                            prob = sortedTuples[1][1]
                    else:
                        result = sortedTuples[0][0]
                        prob = sortedTuples[0][1]
                    if result=='UNKNOWN':
                        result = 'person'  # a default       
                    
                df.loc[df.wordIX == wordIX, 'NcatResult']  = result
                df.loc[df.wordIX == wordIX, 'NcatProb']    = prob


    
def filenamesForNNTag(nnTag, modelInfo, sessionTag):
    pid               = modelInfo[nnTag]['id']  
    
    if isinstance(pid, int):
        modelFn           = '%05d_best_dev' % pid
    else:
        modelFn           = pid

    
    modelCreationDBFn = modelInfo[nnTag]['db']  
    if not sessionTag:
        sessionTag = ''
    z = sessionTag.split('/')
    stag = z[-1]
    testVectorDBFn    = '%s_%stestVectors.db' % (nnTag, stag)
    resultsDBFn       = '%s_%sresults.db'     % (nnTag, stag)
    return (modelCreationDBFn, modelFn, testVectorDBFn, resultsDBFn)

def daisyluSystemEndToEnd(inputFn, sents=None, useNER=True, useCacheIfAvail=True, sessionTag = None, 
                          modelInfo=None, debugSave=False, NoConceptThreshold=0.5, conceptRejectionThreshold=0.0,
                          NEWPrediction=False,  L0OnlyFromFeaturesDB=False, useDistSG=False):
    '''
    Generate list of sentence objects with predicted dataframes from an input text file.
    
    :param inputFn: Text input file with "::tags" including ::snt specification separated by blank lines
    :param sents:   Optional array of pre-processed sentence objects (could include golden info)
    :param useNER:  Use NER output from wikification
    :param sessionTag:  optional Prefix tag for generated files
    :param modelInfo:  Optional structure defining saved nn models and vector/architecture files
    '''
    
    
    #if NEWPrediction:    
    #    keepSense=False
    #else:
    keepSense=True
    modelDBFn=''
    pid=0
        
    # 1. create a standard data frame with one row per word and add to each sentence
    if not sents: 
        sentsRaw = {'test':[]} 
        sentsRaw['test'], _ = readAllAMR(inputFn)
        sents = sentsRaw
    ixList = range(len(sents['test']))
    initializePredictionDataFrames(sents, ixList, NEWPrediction=NEWPrediction)  
    
    # 2a. create vector db from the sentences
    if (useNER):
        nnTag = 'AMRL0'
    else:        
        nnTag = 'AMRL0NoNER'
    (modelCreationDBFn, modelFn, testVectorDBFn, resultsDBFn) = filenamesForNNTag(nnTag, modelInfo, sessionTag)
    wordDF = createVectorsFromDataFrames(sents, 'predictedDFrame', modelCreationDBFn, testVectorDBFn, nnTag,  keepSense=keepSense)        
    # 2b. run SG neural net
    runNetwork('SG',testVectorDBFn, modelFn, resultsDBFn)
    # 2c. add SG nn output to data frames
    addTorchNetworkResults(sents, resultsDBFn, testVectorDBFn, 'AMRL0', 
                           NoConceptThreshold = NoConceptThreshold, 
                           conceptRejectionThreshold = conceptRejectionThreshold)
    if debugSave: pickle.dump( sents, open( 'e2eDebug2.pcl', "wb" ) ) 

    nnTag = 'AMRL0Ncat'
    if nnTag in modelInfo:  # named category is a new option
        (modelCreationDBFn, modelFn, testVectorDBFn, resultsDBFn) = filenamesForNNTag(nnTag, modelInfo, sessionTag)
        createVectorsFromDataFrames(sents, 'predictedDFrame', modelCreationDBFn, testVectorDBFn, nnTag,   keepSense=keepSense, 
                                      L0OnlyFromFeaturesDB=L0OnlyFromFeaturesDB, useDistSG=useDistSG)        
        runNetwork('Cat',testVectorDBFn, modelFn, resultsDBFn)
        addTorchNetworkResults(sents, resultsDBFn, testVectorDBFn, nnTag)
        if debugSave: pickle.dump( sents, open( 'e2eDebugNcat.pcl', "wb" ) ) 

    nnTag = 'AMRL0Args'
    if nnTag in modelInfo:  # named category is a new option
        (modelCreationDBFn, modelFn, testVectorDBFn, resultsDBFn) = filenamesForNNTag(nnTag, modelInfo, sessionTag)
        createVectorsFromDataFrames(sents, 'predictedDFrame', modelCreationDBFn, testVectorDBFn, nnTag,  keepSense=keepSense,
                                    L0OnlyFromFeaturesDB=L0OnlyFromFeaturesDB, useDistSG=useDistSG)        
        runNetwork('Args',testVectorDBFn, modelFn, resultsDBFn)
        addTorchNetworkResults(sents, resultsDBFn, testVectorDBFn, nnTag)
        if debugSave: pickle.dump( sents, open( 'e2eDebug3.pcl', "wb" ) ) 


    nnTag = 'AMRL0Nargs'
    if nnTag in modelInfo:  # named category is a new option
        (modelCreationDBFn, modelFn, testVectorDBFn, resultsDBFn) = filenamesForNNTag(nnTag, modelInfo, sessionTag)
        createVectorsFromDataFrames(sents, 'predictedDFrame', modelCreationDBFn, testVectorDBFn, nnTag,  keepSense=keepSense,
                                    L0OnlyFromFeaturesDB=L0OnlyFromFeaturesDB, useDistSG=useDistSG)        
        runNetwork('Nargs',testVectorDBFn, modelFn, resultsDBFn)
        addTorchNetworkResults(sents, resultsDBFn, testVectorDBFn, nnTag)
        if debugSave: pickle.dump( sents, open( 'e2eDebug4.pcl', "wb" ) ) 


    nnTag = 'AMRL0Attr'
    if nnTag in modelInfo:  # named category is a new option
        (modelCreationDBFn, modelFn, testVectorDBFn, resultsDBFn) = filenamesForNNTag(nnTag, modelInfo, sessionTag)
        createVectorsFromDataFrames(sents, 'predictedDFrame', modelCreationDBFn, testVectorDBFn, nnTag,  keepSense=keepSense,
                                    L0OnlyFromFeaturesDB=L0OnlyFromFeaturesDB, useDistSG=useDistSG)        
        runNetwork('Attr',testVectorDBFn, modelFn, resultsDBFn)
        addTorchNetworkResults(sents, resultsDBFn, testVectorDBFn, nnTag)
        if debugSave: pickle.dump( sents, open( 'e2eDebug5.pcl', "wb" ) ) 

    return sents, wordDF
    


def alignedInputDryrunFlow(amrSents, outFn,  sessionTag,
                           modelInfo=None,
                           useCacheIfAvail=True, 
                           useNER=False, debugSave=False, checkResults=False,
                           NoConceptThreshold=0.65,
                           conceptRejectionThreshold=0.0,
                           forceSubGroupConnectionThreshold=0.35,
                           NEWPrediction=False,
                           L0OnlyFromFeaturesDB=False, 
                           useDistSG=False):   # instead of just ::snt, read from alignments amr, try to use the same sentence boundaries in multi-sent

    sents, wordDF = daisyluSystemEndToEnd(None, sents=amrSents, useNER=useNER, 
                                  sessionTag = sessionTag, 
                                  modelInfo=modelInfo, 
                                  useCacheIfAvail= useCacheIfAvail, 
                                  conceptRejectionThreshold=conceptRejectionThreshold,
                                  debugSave=debugSave, 
                                  NoConceptThreshold=NoConceptThreshold,
                                  NEWPrediction=NEWPrediction,
                                  L0OnlyFromFeaturesDB=L0OnlyFromFeaturesDB, 
                                  useDistSG=useDistSG )

    for i in range(len(sents['test'])):
        s = sents['test'][i]
        print i, s.source['metadata']['id'], s.multiSentIX, s.tokens   
    
    createOutputTextFile(sents, outFn, modelInfo=modelInfo,
                         forceSubGroupConnectionThreshold=forceSubGroupConnectionThreshold)
    if checkResults:
        pickle.dump( sents, open( sessionTag + outFn + '_2.pcl', "wb" ) ) 
    return sents    



if __name__ == '__main__':

    desc = """
       python daisylu_main.py
           """
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-a','--aligned',              help='aligned input',                    action='store_true', default=True)
    
    parser.add_argument('-i','--infile',               help='input file name',                  required=False, default='TINY_amr-bank-struct-v1.6-test.txt')
    parser.add_argument('-o','--outfile',              help='output file name',                 required=False, default='TINY_amr-bank-struct-v1.6-test.amr')
    parser.add_argument('-g','--goldfile',             help='gold file name',                   required=False, default='TINY_amr-bank-struct-v1.6-test')
    parser.add_argument('-t','--tag',                  help='results and temp file tag',        required=False, default='tmp_')
    parser.add_argument('-m','--modelString',          help='modelString, like REFERENCE_MODELS',         required=False, default='REFERENCE_MODELS')
    parser.add_argument('-pid','--pid',                help='pid for AWS',                      required=False, default=-1,  type=int)
    
    parser.add_argument('-nct','--noConceptThreshold', help='no Concept Threshold prob',        required=False, default=0.65, type=float)
    parser.add_argument('-sgt','--subGroupThreshold',  help='sub Group Threshold prob',         required=False, default=0.55, type=float)

    parser.add_argument('-smatch2','--smatch2',         help='smatch2 conversion',              action='store_true', default=True )
    
    args = vars(parser.parse_args())
    pprint (args)
      
    WordRepsFileLocations.init('../data/WORD_LIST.txt')

    pd.set_option('display.width',    1000)
    pd.set_option('display.max_rows', 2000)

    useDistSG=True
    mi = {}
    mi['AMRL0NoNER']  = { 'id': 0,    'db': 'None' }
    mi['AMRL0']       = { 'id': './models/SG.model@./models/SG.weights'         , 'db': 'LDC15_G300ML_Concepts.db' } 
    mi['AMRL0Args']   = { 'id': './models/Args.model@./models/Args.weights'     , 'db': 'LDC15_G300ML_SG_prob_Args.db' } 
    mi['AMRL0Nargs']  = { 'id': './models/Nargs.model@./models/Nargs.weights'   , 'db': 'LDC15_G300ML_SG_prob_Nargs.db' }
    mi['AMRL0Attr']   = { 'id': './models/Attr.model@./models/Attr.weights'     , 'db': 'LDC15_G300ML_SG_prob_Attr.db' } 
    mi['AMRL0Ncat']   = { 'id': './models/Ncat.model@./models/Ncat.weights'     , 'db': 'LDC15_G300ML_SG_prob_Cat.db' } 
    modelInfoDict = {'REFERENCE_MODELS': mi}
                
    outfile1 = args['outfile']
    outfile2 = 'corrected-' + outfile1
    sList={}
      
    sList['test'], _ = readAllAMR(args['infile'])
    sents = alignedInputDryrunFlow(sList, outfile1, 
                           args['tag'], 
                           useNER=True, 
                           modelInfo = modelInfoDict[args['modelString']],
                           conceptRejectionThreshold=0.20,   # <------------------------ New
                           NoConceptThreshold=args['noConceptThreshold'],
                           forceSubGroupConnectionThreshold=args['subGroupThreshold'],    
                           NEWPrediction=True,
                           useDistSG=useDistSG   )

    forceICorefs(sents)
    removeQuantHMMAttrs(sents)
    translateCountryCat(sents)

    createOutputTextFile(sents, outfile2, modelInfo=modelInfoDict[args['modelString']], forceSubGroupConnectionThreshold=args['subGroupThreshold'] )

    if args['goldfile']:
        cmd = getSystemPath('smatchCommand') + ' -r 25 -f %s %s' % ( args['goldfile'], outfile2)
        print cmd
        res = subprocess.check_output(cmd, shell=True)        
        print 'result is ',   res
            
    print 'Done'    
    exit(1)

 
        
         
