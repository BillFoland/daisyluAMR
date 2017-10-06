
import numpy as np
import pandas as pd
import pickle
from pprint import pprint as p
import os
import sqlite3
from keras.utils import np_utils


#
#
# Embedding mask ix (aka pad ix) is tacked onto the end currently.
# To make it consistent, make all mask IX = 0 by prepending 0 and keeping lua indices.
#
# Distinguish masking from padding
# make number of BLSTM layers and output Layers a parameter
# investigate target vectors with all 'O'
#


class AMRDataGenerator(object):

    mask = {}
    targetFeature =  ''
    featureOrder = []
    masking = {} 
    
    def __init__(self, fn, sType,  batchSize, startIX=0, endIX=-1, maxItems=None, forceHardSG=False):
        self.maxSentenceLength = 100
        self.forceHardSG = forceHardSG
        self.sType = sType
        self.fn = fn
        self.conn = sqlite3.connect(self.fn)
        self.conn.text_factory = str   # this is new...
        self.features = self.readAMRVectorFeatureInfo( )
        self.setFeatureBasedProperties()  # NN specific
        self.wordEmbedFn   = self.features['words']['filename']
        if 'glove' in self.wordEmbedFn.lower():
            self.vectorEmbedFn = self.wordEmbedFn.replace('Word', 'Vector')
        else:
            self.vectorEmbedFn = '../data/senna_embeddings.txt' 
       
        self.fCount   = self.getDatasetLengths(maxItems)   
        self.pad={'distSG': self.getAMRDBFeatureSize()}
        self.vec, self.target, self.sentenceIX = self.readAMRVectorDatabase( 0, self.fCount[sType] )
            
        self.features['target']={'width': self.target.shape[2] }

        self.conn.close()
        
        
        self.startIX   = startIX
        self.currentIX = startIX
        self.endIX     = endIX
        self.batchSize = batchSize


    def getAMRDBFeatureSize(self):
        count=0
        self.conn = sqlite3.connect(self.fn)
        cur = self.conn.cursor()
        cmd = "SELECT name FROM sqlite_master WHERE type='table' AND name = 'DBFeatures'"
        cur.execute(cmd)
        for row in cur:
            cmd = "SELECT DISTINCT type FROM DBFeatures"
            cur.execute(cmd)
            dbFeatureTypes = []
            for row in cur:
                dbFeatureTypes.append(row[0])  
            if len(dbFeatureTypes):
                cmd = "SELECT count(*) FROM DBFeatures WHERE type = '%s'" % dbFeatureTypes[0]
                cur.execute(cmd)
                keys = [z[0] for z in cur.description]
                for row in cur:
                    count = int(row[0])
        return count + 1  # first row is mask, last is PADDING  
    
    def readAMRDBFeatureInfo(self):
        
        #
        # extra column added to end (131 +1) for the PAD bit
        # extra row added at beginning for mask, all zeros.
        # extra row added at end for PAD bit active, it is inactive everywhere else.
        #
        self.conn = sqlite3.connect(self.fn)
        cur = self.conn.cursor()
     
        cmd = "SELECT DISTINCT type FROM DBFeatures"
        cur.execute(cmd)
        dbFeatureTypes = []
        for row in cur:
            dbFeatureTypes.append(row[0])  
        
        featureNPArray = np.array([])
        if len(dbFeatureTypes):
            cmd = "SELECT count(*) FROM DBFeatures WHERE type = '%s'" % dbFeatureTypes[0]
            cur.execute(cmd)
            keys = [z[0] for z in cur.description]
            for row in cur:
                count = int(row[0])
    
            cmd = "SELECT ix, CSV FROM DBFeatures WHERE type = '%s'" % dbFeatureTypes[0]
            cur.execute(cmd)
            keys = [z[0] for z in cur.description]
            for row in cur:
                d = dict(zip(keys,row))
                npA = np.array([float(x) for x in d['CSV'].strip().split(',')]  ) 
                if featureNPArray.shape[0]==0:
                    featureNPArray = np.zeros( (count+2, len(npA) ) )     # count+2 to allow for mask at index 0, padding at end
                featureNPArray[d['ix']] = npA
            featureNPArray[-1][-1]      = 1.0                             # set the PAD bit in last element
        self.conn.close()
        

        return featureNPArray


    def readAMRVectorFeatureInfo(self):
        cur = self.conn.cursor()
     
        cmd = "SELECT * FROM Tokens"
        cur.execute(cmd)
        keys = [z[0] for z in cur.description]
        features = {}
        for row in cur:
            d = dict(zip(keys,row))
            tp = d['type']
            token = d['token']
            ix = d['ix']                               
            if not tp in features:
                if tp == self.targetFeature:
                    features[tp] = { 'tokens':[], 't2i':{} } # no masking token for the target
                else:    
                    features[tp] = {'tokens':['MASKIX0'], 't2i':{'MASKIX0':0}}
            features[tp]['tokens'].append(token)
            features[tp]['t2i'][token]=ix
            if (ix != len(features[tp]['tokens'])):
                assert('Error in read tokens from db')
                
        cmd = "SELECT * FROM WDFArch ORDER BY ix"
        cur.execute(cmd)
        keys = [z[0] for z in cur.description]
        for row in cur:
            d = dict(zip(keys,row))
            print 'NAME:', d['name']
            if not d['name'] in features:
                features[d['name']] = {}
            features[d['name']].update(d)
            features[d['name']]['shape'] = ( d['size']+1,  d['width'] ) # added space for mask vector
    
        for tp in self.featureOrder:
            #print features[tp]

            if (features[tp]['clone'] == None):
                if features[tp]['filename'] != None:
                    if features[tp]['filename'][:3]== 'DB:':
                        continue
                if not 'tokens' in features[tp]:
                    continue
                features[tp]['shape'] = ( len(features[tp]['tokens']) + 1,  features[tp]['width'] ) # added space for mask vector
            #else:
                #cloneTag = features[tp]['clone']
                #features[tp]['shape'] = ( len(features[cloneTag]['tokens']),  features[cloneTag]['width'] ) 
                
        return features      
    


    def readAMRVectorDatabase(self, start=0, end=2 ):
        db = self.conn
        features = self.features
        fDict = {}
        allDict = {}
        localDict = {}
        for f in self.featureOrder:
            fDict[f] = np.ones( (end-start, self.maxSentenceLength ) , dtype='int32') * (self.masking[f])
            allDict[f] = []
        targets = np.ones( (end-start, self.maxSentenceLength, len(self.features[self.targetFeature]['tokens']) ) ) * (self.masking['target'])
        sentenceIX = np.zeros( (end-start, 1) ) 
        cur = db.cursor()
        for zeroBased in range(start, end):
            pnum = zeroBased+1
            # 1 is the first vector.
            cmd = "SELECT ix, type, sentIX,  pWordIX,  targetCSV FROM Items WHERE ix = %d AND type = '%s'"  % (pnum, self.sType)
            cur.execute(cmd)
            keys = [z[0] for z in cur.description]
            for row in cur:
                d = dict(zip(keys,row))
                npA = np.array([int(x) for x in d['targetCSV'].strip().split(',')])[:self.maxSentenceLength]
                npA = npA - 1 #  X adjustment for target, since we don't inject a zero mask in a lookup table for target
                categorical = np_utils.to_categorical(npA, len(self.features[self.targetFeature]['tokens']))
                numWords = len(npA) 
                targets[zeroBased-start][-numWords:] = categorical
                pWordIX = d['pWordIX']
                if d['pWordIX']:
                    pWordIX = d['pWordIX'] -1 # adjust for zero based from lua
                sentIX  = d['sentIX']
                
            sentenceIX[zeroBased] = sentIX    
            cmd = "SELECT ix, type, fType,  fCSV FROM Sentences WHERE ix = %d AND type = '%s'" % (sentIX, self.sType)
            cur.execute(cmd)
            keys = [z[0] for z in cur.description]
            for row in cur:
                d = dict(zip(keys,row))
                if not d['fType'] in allDict:
                    allDict[d['fType']] = []
                allDict[d['fType']].append(d['fCSV'])
                if d['fType'] in self.featureOrder:
                    npA = np.array([int(x) for x in d['fCSV'].strip().split(',')])[:self.maxSentenceLength]
                    # do the right shift later: fDict[d['fType']][zeroBased-start][-len(npA):] = npA
                    localDict[d['fType']] = npA  
              
                    # {'ix': 14, 'name': 'L0ctxP3', 'clone': 'distSG', 'ptrName': 'itemP', 'filename': None, 
                    # 'width': 131, 'fType': 'distSG', 'learningRate': 0.0, 'offset': 0, 'size': 401185}      
                    
            for i,f in enumerate(self.featureOrder):
                    
                wf =  []
                ptrName = self.features[f]['ptrName']
                fSource = self.features[f]['clone']
                if fSource == None:
                    fSource = f
                offset  = self.features[f]['offset']
 
                for w in range(numWords):
                    if ptrName == 'wordP':                                                # < ---- PADDING
                        pointer = w
                        pointer = pointer + offset
                        if (pointer < 0) or (pointer >= numWords):
                            if not 't2i' in self.features[fSource]:
                                #print 't2i not in', fSource
                                pad = self.pad[fSource]   # <-- note pad, not mask
                            else:    
                                pad   = self.features[fSource]['t2i']['PADDING']
                            wf.append(pad)
                        else:
                            wf.append(localDict[fSource][pointer])
                        
                    elif ptrName == 'itemP':                                              # < ---- PADDING
                        pointer = pWordIX
                        pointer =  pointer + offset
                        if (pointer < 0) or (pointer >= numWords):
                            if not 't2i' in self.features[fSource]:
                                #print 't2i not in', fSource
                                pad = self.pad[fSource]   # <-- note pad, not mask
                            else:    
                                pad   = self.features[fSource]['t2i']['PADDING']
                            wf.append(pad)
                        else:
                            wf.append(localDict[fSource][pointer])
    
 
                    elif ptrName=='deltaP':                                               # < ---- PADDING
                        pointer = w - pWordIX
                        pointer =  pointer + offset
                        size = self.features[fSource]['size']
                        middleIX = size/2 
                        tableIX = middleIX + pointer
                        if tableIX <0:
                            tableIX = 0
                        if tableIX >= size:
                            tableIX = size-1
                        wf.append(tableIX+1)
                    else:
                        print 'ERROR'
                fDict[f][zeroBased-start][-len(wf):] = np.array(wf, dtype='int32')
                        
        if self.forceHardSG:
            # read distSG,
            # argMax it
            # substitute HardSG for distSG in vectors    
            distSG_embedding_matrix = self.readAMRDBFeatureInfo()
            txIX = np.array([np.argmax(x)+1 for x in distSG_embedding_matrix]) # the plus one makes room for the mask at zero...
            txIX[0] = 0
            for f in ['distSG', 'L0Pred', 'L0ctxP1', 'L0ctxP2', 'L0ctxP3', 'L0ctxP4', 'L0ctxP5']:
                fDict[f] = txIX[fDict[f]]
        self.features['SG']={'shape':(len(self.features['L0']['tokens']),10)}
    
            
            
        ret = []
        for f in self.featureOrder:
            ret.append(fDict[f])  # no index adjustment
        
        return ret, targets, sentenceIX


    def getDatasetLengths(self, maxItems):
        count = {}
        cur = self.conn.cursor()

        for sType in [ 'training', 'test', 'dev' ]:
            "SELECT COUNT(*) FROM Items WHERE type = '%s'", sType 
            cur.execute("SELECT COUNT(*) FROM Items WHERE type = '%s'" % sType)
            (ret) = cur.fetchone()
            count[sType] = ret[0]
        count['train'] = count['training']
        if maxItems:
            for k in count.keys():
                if count[k] > maxItems:
                    count[k] = maxItems
            
        return count
        
         
    def numberOfItems(self):
        return self.fCount[self.sType] 


    def setCurrentIX(self, ix):  
        self.currentIX = ix


    def getTargets(self, numSamples):
        return self.target

         
    def generate(self):
        self.currentIX = self.startIX
        if self.endIX == -1:
            nbSamples = self.vec[0].shape[0] 
        else:
            nbSamples = self.endIX
        print 'There are %d items in the data set' % self.vec[0].shape[0]
        if not self.vec[0].shape[0]:
            exit(1)
        
        while(1):       
            if (self.currentIX + self.batchSize) <= nbSamples:  # nbSamples is the length, so it is actual max IX + 1
                vec    = [self.vec[i][self.currentIX:self.currentIX + self.batchSize] for i in range(len(self.vec)) ]
                target = self.target[self.currentIX: self.currentIX + self.batchSize]
                #print 'DEBUG', vec[0].shape, self.sType, self.currentIX, self.currentIX + self.batchSize
                self.currentIX += self.batchSize  
                yield ( vec, target )
            else: # Wraparound
                vec     = [ self.vec[i][self.currentIX:nbSamples]  for i in range(len(self.vec)) ]
                target  =  self.target[self.currentIX:nbSamples]  
                self.currentIX = self.startIX  
                while target.shape[0] < self.batchSize:
                    p1_end   = np.minimum( self.currentIX + nbSamples, self.currentIX + (self.batchSize-target.shape[0]) )
                    vec     = [ np.append( vec[i] , self.vec[i][self.currentIX:p1_end] , axis=0) for i in range(len(self.vec)) ]
                    target  = np.append(  target , self.target[self.currentIX:p1_end] , axis=0) 
                    #print '  EDGE DEBUG ', vec[0].shape, self.currentIX, p1_end
                    if p1_end == nbSamples:
                        self.currentIX = self.startIX
                    else:
                        self.currentIX = p1_end
                yield ( vec, target )



    @staticmethod
    def getGeneralArch(fn): 
        db = sqlite3.connect(fn)
        cur = db.cursor()
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
        return generalArch 
 
    def writeAMRResultsDatabase(self, fn, yProbs, targets):
        
        
        def npArrayToCSV(a):
            st = u''
            if len(a.shape)==2: # 2d 
                for r in range(a.shape[0]):
                    row = a[r]
                    st += ','.join([str(x) for x in row]) + '#'
                st = st[:-1] # get rid of trailing #
            else:
                st += ','.join([str(x) for x in a]) 
            return st

        db = sqlite3.connect(fn)
        c = db.cursor()
        c.execute(  "DROP TABLE IF EXISTS Items" )
        c.execute(  'CREATE TABLE IF NOT EXISTS Items( ix int, type text, resultVector text, logProbVector text, distSG text, target text, sm int, sc int, rc int, score float, PRIMARY KEY (ix, type))' )
        db.commit()
    
        """
        LUA code reference
        
        targetString = tensorToCSV(target)
    
      cmd = string.format("INSERT INTO Items (ix, type, resultVector, logProbVector, distSG, target, sm, sc, rc, score) VALUES (%d, '%s', '%s', '%s', '%s', '%s', %d, %d, %d, %f)" ,
           ix, dbTable['sType'], tensorToCSV(result), floatTensorToCSV(lpv:transpose(1,2)), distSGString, targetString, sm, sc, rc, score) 
      res = assert (dbTable['db']:execute(cmd)  )  
      
      
      python ref
          db.execute("INSERT INTO Predicates (ix, sentenceLen, sentenceIX, target, predPos, pred, sense, ref_si, ref_pi) VALUES ( ?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                           (predIX, len(ix[i]), sentIX, csvTarget, pred['predpos']+1, pred['pred'], pred['sense'], sentIX, pi+1 ))
    
        """
        
        labelCount= targets.shape[2]
        y_pred = np.argmax(yProbs, axis=2) 
        MLTarget = np.argmax(targets, axis=2)
        CM = np.zeros( (labelCount, labelCount) )
        sm = 0
        sc = 0
        rc = 0
        indexForO = 0 # target does not need a masked vector
        for i in range(len(MLTarget)):
            item_sm = 0
            item_sc = 0
            item_rc = 0
            firstIX=-1
            for w in range(len(MLTarget[i])):
                if sum(targets[i][w]) >= 0.99:
                    firstIX=w
                    break
            for w in range(firstIX, len(MLTarget[i])):
                if sum(targets[i][w]) >= 0.99:
                    x = MLTarget[i][w]
                    y = y_pred[i][w]
                    if x !=indexForO:
                        rc += 1 # ref
                        item_rc += 1 # ref
                        if x==y:
                            sm+=1 # match
                            item_sm+=1 # match
                    if y != indexForO:
                        sc += 1 # sys   
                        item_sc += 1 # sys   
                    CM [x][y]+=1
            item_precision = 0
            item_recall = 0 
            item_f1 = 0
            if (item_sc > 0) and (item_sm > 0) and (item_rc > 0):
                item_precision = item_sm/(item_sc*1.0)
                item_recall    = item_sm/(item_rc*1.0)
                item_f1 = 2.0/( 1.0/item_precision + 1.0/item_recall)
            
                
            logProbs=  np.log(yProbs[i][firstIX:]) # getting 0 out of this for very close to 1...
            logProbString = npArrayToCSV(logProbs)  
            resultString  = npArrayToCSV(y_pred[i][firstIX:] + 1 )   # lua tags start at 1
            targetString  = npArrayToCSV(MLTarget[i][firstIX:] + 1 ) # lua tags start at 1
            db.execute("INSERT INTO Items (ix, type, resultVector, logProbVector, distSG, target, sm, sc, rc, score) VALUES ( ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                               (i+1,  self.sType, resultString, logProbString, '', targetString, item_sm, item_sc, item_rc, item_f1))
            db.commit()
        db.close()
    
        precision = 0
        recall = 0 
        f1 = 0
        if (sc > 0) and (sm > 0) and (rc > 0):
            precision = sm/(sc*1.0)
            recall    = sm/(rc*1.0)
            f1 = 2.0/( 1.0/precision + 1.0/recall)
        cString='\n   '
        for r in range(np.minimum(20,CM.shape[0])):
            for c in range(np.minimum(20,len(CM[r]))):
                if CM[r][c] > 0:
                    cString += 'X '
                else:        
                    cString += '. '
            cString += "\n   "
        df = pd.DataFrame(CM)
        names = self.features[self.targetFeature]['tokens']
        df.columns = names
        df['name'] = names
        df = df.set_index('name')
        return df, sm, rc, sc, precision, recall, f1, cString 
          




class SGGenerator(AMRDataGenerator):
    targetFeature =  'L0'
    
    def setFeatureBasedProperties(self):
        self.featureOrder = ['words','caps', 'suffix', 'ner' ]
        self.masking = {'words':0, 'caps':0, 'suffix':0, 'ner':0, 'target':0 } 


class ArgsGenerator(AMRDataGenerator):
    targetFeature =  'args'
    
    def setFeatureBasedProperties(self):
        if 'distSG' in self.features.keys():
            sgRep = 'distSG'
        else:
            sgRep = 'L0'
        self.featureOrder = ['words','caps', 'suffix', 'Pred',
                        'ctxP1', 'ctxP2', 'ctxP3', 'ctxP4', 'ctxP5', 
                         sgRep, 'L0Pred',
                        'L0ctxP1', 'L0ctxP2', 'L0ctxP3', 'L0ctxP4', 'L0ctxP5',
                        'regionMark'
                       ]
        self.masking = {x:0 for x in self.featureOrder}
        self.masking['target'] = 0
    
    def printVectors(self):
        self.printVectorsArgsNargs()

class ArgsHardSGGenerator(AMRDataGenerator):
    targetFeature =  'args'

    def setFeatureBasedProperties(self):
        if 'distSG' in self.features.keys():
            sgRep = 'distSG'
        else:
            sgRep = 'L0'
        self.featureOrder = ['words','caps', 'suffix', 'Pred',
                        'ctxP1', 'ctxP2', 'ctxP3', 'ctxP4', 'ctxP5', 
                         sgRep, 'L0Pred',
                        'L0ctxP1', 'L0ctxP2', 'L0ctxP3', 'L0ctxP4', 'L0ctxP5',
                        'regionMark'
                       ]
        self.masking = {x:0 for x in self.featureOrder}
        self.masking['target'] = 0
        


class NargsGenerator(AMRDataGenerator):
    targetFeature =  'nargs'
    
    def setFeatureBasedProperties(self):
        if 'distSG' in self.features.keys():
            sgRep = 'distSG'
        else:
            sgRep = 'L0'
        self.featureOrder = ['words','caps', 'suffix', 'Pred',
                        'ctxP1', 'ctxP2', 'ctxP3', 'ctxP4', 'ctxP5', 
                         sgRep, 'L0Pred',
                        'L0ctxP1', 'L0ctxP2', 'L0ctxP3', 'L0ctxP4', 'L0ctxP5',
                        'regionMark'
                       ]
        self.masking = {x:0 for x in self.featureOrder}
        self.masking['target'] = 0
        
    def printVectors(self):
        self.printVectorsArgsNargs()

class AttrGenerator(AMRDataGenerator):
    targetFeature =  'attr'

    def setFeatureBasedProperties(self):
        if 'distSG' in self.features.keys():
            sgRep = 'distSG'
        else:
            sgRep = 'L0'   
        self.featureOrder = ['words','caps', 'suffix', sgRep ]
        
        self.masking = {x:0 for x in self.featureOrder}
        self.masking['target'] = 0
    
    def printVectors(self):
        
        distSG_embedding_matrix = self.readAMRDBFeatureInfo()
        embedding_matrix, maskIX = self.getEmbeddings()
        
        for vcount, (vec,target) in enumerate(self.generate()):       
            sample = 0
            if (False):
                for i,f in enumerate(vec):
                    print 
                    print self.featureOrder[i]
                    print i, f 
            d={}
            d['words']  = [self.features['words']['tokens'][x]  for x in vec[0][sample] if x != 0]    
            d['caps']   = [self.features['caps']['tokens'][x]   for x in vec[1][sample] if x != 0]    
            d['suffix'] = [self.features['suffix']['tokens'][x] for x in vec[2][sample] if x != 0]    
            
            distSG = [  distSG_embedding_matrix[x] for x in  vec[3][sample] if x != 0 ]    
            tMaxIX = [ np.argmax(x) for x in distSG ]    
            d['distSG']  = [self.features['L0']['tokens'][x+1] for x in [ np.argmax(x) for x in [  distSG_embedding_matrix[x] for x in  vec[3][sample]  if x != 0 ] ]  ]    
            
            tMaxIX = [ np.argmax(x) for x in target[sample] ]        
            d['target'] = [self.features[self.targetFeature]['tokens'][x] for x in tMaxIX[-len(d['words']):] ]    
            
            df = pd.DataFrame(d)
            df= df[ self.featureOrder + ['target'] ]
            print df
            
            if vcount > 20:
                return

class CatGenerator(AMRDataGenerator):
    targetFeature =  'ncat'
    
    def setFeatureBasedProperties(self):
        if 'distSG' in self.features.keys():
            sgRep = 'distSG'
        else:
            sgRep = 'L0'   
        self.featureOrder = ['words','caps', 'suffix', sgRep, 
                        'wcat',
                        'wcat1',  'wcat2',  'wcat3',  'wcat4',  'wcat5',  'wcat6',  'wcat7',     
                       ]
        self.masking = {x:0 for x in self.featureOrder}
        self.masking['target'] = 0
    
      
if __name__ == '__main__':
    
    pd.set_option('display.width',    1000)
    pd.set_option('display.max_rows', 2000)
     
    exit(1)






