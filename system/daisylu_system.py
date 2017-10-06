
import os
import sys
import pickle
import pandas as pd
import numpy as np
import hashlib
import os.path

from daisylu_config import *
from daisylu_vectors import *
from sentences import *
from daisylu_output import *
""       
                   

def addWikificationToDFrames(sents, sTypes, sentenceAttr):   
    # need to split this up into manageable file sizes, the wikifier dies with out of memory error currently

    maxPartitionLen=200000
    resultsDir = getSystemPath('daisyluPython') + '/wikificationData'
    
    # if the working directory does not exist, create it.
    # './wikificationData/input/test0.txt'
    if not os.path.exists(resultsDir):
        os.makedirs(resultsDir)
    if not os.path.exists(resultsDir+'/input'):
        os.makedirs(resultsDir+'/input')
    if not os.path.exists(resultsDir+'/output'):
        os.makedirs(resultsDir+'/output')
    
    
    for sType in sTypes:
        partitions = [   {'textString':'', 'charMapper':{}}  ]
        for sentIX in range(len(sents[sType])):  
            if len(partitions[-1]['textString']) > maxPartitionLen:
                partitions.append( {'textString':'', 'charMapper':{}})
                print 'THERE ARE NOW %d PARTITIONS' % len(partitions)
                print '====================================================================================================================='
                print '====================================================================================================================='
                print

            sentence = sents[sType][sentIX]
            if not sentIX % 100: print 'addWikificationToDFrames', sType, sentIX
            if not hasattr(sentence, sentenceAttr):
                continue
            sdf = getattr(sentence, sentenceAttr)
            if sdf.empty:
                continue    
            sdf['NERForm']    = ''
            sdf['NERLabel']   = 'O'
            sdf['WKCategory'] = ''
            sdf['WKLink']     = ''
            sdf['WKLinker']   = np.nan
            sdf['WKRanker']   = np.nan
            
            df = sdf[['wordIX','words','txFunc','txBIOES','nameCategory','wiki']].copy()
            df['type'] = sType
            df['sentIX'] = sentIX
            df['allTStart'] = -1
            df['allTEnd'] = -1
            for i,t in enumerate(sentence.tokens):
                startOffset = len(partitions[-1]['textString'])
                partitions[-1]['textString'] += t
                endOffset = len(partitions[-1]['textString'])
                if (any(df.wordIX == i)):
                    df['allTStart']=startOffset 
                    df['allTEnd']=endOffset 
                partitions[-1]['charMapper'][startOffset] = (sentIX, i, t)
                partitions[-1]['textString'] += ' '
            partitions[-1]['textString'] += '\n\n'

        allText = ''
        for x in partitions:
            allText += x['textString']
        m = hashlib.md5()
        m.update(allText)
        md5 = m.hexdigest()        
        print md5    
        cacheFn = 'wikificationData/' + md5 + '.pcl'
            
        if not os.path.isfile(cacheFn):   # calculate and archive the info, use it later if the same set of sentences is called for 
            wconfigs = []
            wconfigs.append({
                                'config'   : 'configs/STAND_ALONE_NO_INFERENCE.xml',
                                'inputFn'  : resultsDir + '/input/test%d.txt',
                                'outputDn' : '%s/output/' % resultsDir,
                             })
            info = { 'NER':{}, 'wiki':{} }
            
        
            # partitions = pickle.load( open( 'wikiPartitions.pcl' ) ) 
            """
            If you're using this system, please cite the paper.
            
            Relational Inference for Wikification
            Xiao Cheng and Dan Roth
            EMNLP 2013
            """
            for p, partition in enumerate(partitions):
                for wtype in wconfigs:
                    tfile = open(wtype['inputFn'] % p, 'wb')
                    tfile.write(partition['textString'])
                    tfile.close()
                            
                    direc     = getSystemPath('Wikifier2013') 
                    config    = wtype['config']
                    inputFn   = wtype['inputFn'] % p
                    outputDn  = wtype['outputDn'] 
                    stencil   = '/usr/bin/java -Xmx10G -jar dist/wikifier-3.0-jar-with-dependencies.jar -annotateData %s %s false %s'    
                    cmd = stencil % (inputFn, outputDn, config)
                    print cmd
                    errorCode = os.system('cd %s; %s' % (direc, cmd)   )
                    if errorCode:
                        raise ValueError('ERROR!\n    non zero error code %d' % errorCode)
                        exit(1)
            
            
                        
            import xmltodict
            from bs4 import BeautifulSoup
            
            for p, partition in enumerate(partitions):
                print 'Partition %d' % p
                charMapper = partitions[p]['charMapper']
                html = open('%s/output/' % resultsDir + '/test%d.txt.NER.tagged' % p).read()
                parsed_html = BeautifulSoup(html, "lxml")
                ner={ 'start':[], 'end':[], 'form':[], 'label':[]}
                for item in parsed_html.find_all('start'):
                    ner['start'].append(int(item.text))
                for item in parsed_html.find_all('end'):
                    ner['end'].append(int(item.text))
                for item in parsed_html.find_all('form'):
                    ner['form'].append(item.text)
                for item in parsed_html.find_all('label' ):
                    ner['label'].append(item.text)
                    
                for i in range(len(ner['start'])):
                    if not i % 100: print 'ner', i
                    tset = set()
                    for z in range(ner['start'][i],ner['end'][i]):
                        if z in charMapper:
                            tset.add(charMapper[z])
                    for trip in list(tset):
                        (six, wix, _) = trip                     
                        if not six in info['NER']:
                            info['NER'][six] = { 'NERForm':{}, 'NERLabel':{} }
                        info['NER'][six]['NERForm'][wix]       =  ner['form'][i]
                        info['NER'][six]['NERLabel'][wix]      =  ner['label'][i]
                
                
                
                with open('%s/output/' % resultsDir + '/test%d.txt.wikification.tagged.full.xml' % p) as fd:
                    obj = xmltodict.parse(fd.read())
                    if obj['WikifierOutput']['WikifiedEntities']:
                        entities = obj['WikifierOutput']['WikifiedEntities']['Entity']
                        for entity in entities:
                            #entityText = entity['EntitySurfaceForm']       
                            entityStartOffset = int(entity['EntityTextStart'])        
                            entityEndOffset   = int(entity['EntityTextEnd'])  
                            linkerScore       = float(entity['LinkerScore'])       
                            rankerScore       = float(entity['TopDisambiguation']['RankerScore'])              
                            wikiTitle         = entity['TopDisambiguation']['WikiTitle']            
                            attributes        = entity['TopDisambiguation']['Attributes']              
                
                            #print   entityText, entityStartOffset, entityEndOffset, textString[entityStartOffset:entityEndOffset]
                            tset = set()
                            for z in range(entityStartOffset,entityEndOffset+1):
                                if z in charMapper:
                                    tset.add(charMapper[z])
                            
                            for trip in list(tset):
                                (six, wix, _) = trip 
    
                                if not six in info['wiki']:
                                    info['wiki'][six] = { 'WKCategory':{}, 'WKLink':{}, 'WKLinker':{}, 'WKRanker':{} }
                                info['wiki'][six]['WKCategory'][wix]       = attributes
                                info['wiki'][six]['WKLink'][wix]           = wikiTitle
                                info['wiki'][six]['WKLinker'][wix]         = linkerScore
                                info['wiki'][six]['WKRanker'][wix]         = rankerScore

            pickle.dump( info, open( cacheFn, "wb" ) ) 
   
        else:
            info = pickle.load( open( cacheFn, "rb" ) ) 
   
        for six in info['NER']:
            sentence = sents[sType][six]
            if not hasattr(sentence, sentenceAttr):
                continue
            sdf = getattr(sentence, sentenceAttr)                   
            for wix in info['NER'][six]['NERForm']:
                sdf.loc[  (sdf.wordIX == wix), 'NERForm']     = info['NER'][six]['NERForm'][wix]  
                sdf.loc[  (sdf.wordIX == wix), 'NERLabel']    = info['NER'][six]['NERLabel'][wix]  
        for six in info['wiki']:
            sentence = sents[sType][six]
            if not hasattr(sentence, sentenceAttr):
                continue
            sdf = getattr(sentence, sentenceAttr)                   
            for wix in info['wiki'][six]['WKCategory']:
                sdf.loc[  (sdf.wordIX == wix), 'WKCategory']  = info['wiki'][six]['WKCategory'][wix]
                sdf.loc[  (sdf.wordIX == wix), 'WKLink']      = info['wiki'][six]['WKLink'][wix]
                sdf.loc[  (sdf.wordIX == wix), 'WKLinker']    = info['wiki'][six]['WKLinker'][wix]
                sdf.loc[  (sdf.wordIX == wix), 'WKRanker']    = info['wiki'][six]['WKRanker'][wix]  
 
                    


def initializePredictionDataFrames(sents, ixList=None, NEWPrediction=False):
    if not ixList:
        ixList = range(len(sents['test']))

    for sentIX in ixList:  
        sentence = sents['test'][sentIX]
        tagList = getSentenceDFTagList()
        if NEWPrediction:
            tagList += ['NEWPrediction']
        df = pd.DataFrame( columns=tagList )
        if not (sentIX %1000):
            print 'initializing pred frome ', sentIX
        df['wordIX']  = range(len(sentence.tokens))
        df['sentIX']  = sentIX
        df['words']   = sentence.tokens    
        df['txBIOES'] = 'O'      
        sentence.predictedDFrame = df 
    
    addWikificationToDFrames(sents, ['test'], 'predictedDFrame') 
    print 'CLIPPING ALL SENTENCES TO LENGTH 100'
    for sentIX in ixList:  
        sentence = sents['test'][sentIX]
        sentence.predictedDFrame = sentence.predictedDFrame[sentence.predictedDFrame['wordIX'] < 100]
    
    
def createVectorsFromDataFrames(sents, sentenceAttr, dbf, dbtf, systemName, keepSense=True,  L0OnlyFromFeaturesDB=False, useDistSG=False ):
    wordDF = []
    dbfn = getSystemPath('daisylu') + 'data/%s' % dbf
    dbTestFn = getSystemPath('daisylu') + 'data/%s' %  dbtf

    merged = mergeSentenceDataFrames(None, ['test'], None,  sents=sents, sentenceAttr=sentenceAttr)
    if systemName== 'AMRL0NoNER':
        createAMRL0Vectors(None, dbTestFn,  100.0, keepSense, sTypes=['test'], vectors=merged, featuresDB=dbfn, maxSents=None, useNER=False)
    elif systemName== 'AMRL0':
        wordDF = createAMRL0Vectors(None, dbTestFn,  100.0, keepSense, sTypes=['test'], vectors=merged, featuresDB=dbfn, maxSents=None )
    elif systemName== 'AMRL0Args':
        createAMRL0ArgVectors(None, dbTestFn,  100.0, keepSense, sTypes=['test'], vectors=merged, featuresDB=dbfn, maxSents=None, 
                              L0OnlyFromFeaturesDB=L0OnlyFromFeaturesDB, useDistSG=useDistSG )
    elif systemName== 'AMRL0Nargs':
        createAMRL0NargVectors(None, dbTestFn,  100.0, keepSense, sTypes=['test'], vectors=merged, featuresDB=dbfn, maxSents=None, 
                               L0OnlyFromFeaturesDB=L0OnlyFromFeaturesDB, useDistSG=useDistSG )
    elif systemName== 'AMRL0Attr':
        createAMRL0AttrVectors(None, dbTestFn,  100.0, keepSense, sTypes=['test'], vectors=merged, featuresDB=dbfn, maxSents=None, 
                               L0OnlyFromFeaturesDB=L0OnlyFromFeaturesDB, useDistSG=useDistSG )
    elif systemName== 'AMRL0Ncat':
        createAMRL0NcatVectors(None, dbTestFn,  100.0, keepSense, sTypes=['test'], vectors=merged, featuresDB=dbfn, maxSents=None,
                               L0OnlyFromFeaturesDB=L0OnlyFromFeaturesDB, useDistSG=useDistSG )
    else:
        assert('error, invalid system name')
    return wordDF

def runKerasNetwork(networkType, vectorDBFn, modelFn, resultsFn, sType='test'):
    direc = getSystemPath( 'NNModels' )
    mm, ww = modelFn.split('@')
    cmd = getSystemPath( 'python' )
    cmd   = cmd +' AMR_NN_Forward.py  -v %s -m %s -w %s -r %s -s %s' % ('../data/' + vectorDBFn, mm, ww, '../results/' + resultsFn, sType)    
    print direc, cmd
    print direc, cmd
    print direc, cmd
    errorCode = os.system('cd %s; %s' % (direc, cmd)   )
    if errorCode:
        raise ValueError('ERROR!\n    non zero error code %d' % errorCode)
        exit(1)
    if errorCode:
        print vectorDBFn, modelFn, resultsFn, sType
        raise ValueError('ERROR!\n    non zero error code %d' % errorCode)
        exit(1)

def runNetwork(networkType, vectorDBFn, modelFn, resultsFn, sType='test'):
    if '@' in modelFn:
        runKerasNetwork(networkType, vectorDBFn, modelFn, resultsFn, sType)
    else:
        assert('error, Torch networks are no longer supported')

        