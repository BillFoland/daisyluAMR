import os
import sys
import pickle
#from nltk.corpus.reader.wordnet import Lemma
reload(sys)
sys.setdefaultencoding('utf-8')  # @UndefinedVariable

import subprocess
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
from colorama import Style
import traceback
import logging
from daisylu_config import getSystemPath

    

from nltk.stem.wordnet import WordNetLemmatizer
lmtzr=WordNetLemmatizer()#create a lemmatizer object

import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
from nltk.tokenize import sent_tokenize, word_tokenize


class Sentence(object):
    # separate out alignments and tokens here,
    # the rest is reference only, and will point to the multi-sentence data, for example
    # so use alignments, amrp, and tokens from here on out.


    
    def __init__(self, alignments, parsedAMR,  tok, amr_raw, amr_line, metadata, multiSentIX=0 ):
        self.source    = {'amrRaw':None, 'amrText':None, 'metadata':None, 'multiParse':None, 'multiTokens':None, 'multiAligments':None}
        self.pcfg       = None
        self.info       = None
        self.txSpec     = None
        self.docMeta    = {}  # documentation metadata, can be displayed in PDF for example
        self.referenceDFrame = pd.DataFrame()
        self.predictedDFrame = pd.DataFrame()
        self.amrp = parsedAMR
        
        AMRCorpusTranslate={ '@-@':'-',  '@/@':'/',  '."':'.',  '",':',',   '".':'.',   '@:@':':',    
                           '@-':'-',   ')?':'?',   '!!':'!',  '??':'?',   '"?':'?',   '):':')'   }   
        for i,t in enumerate(tok):
            if t in AMRCorpusTranslate:
                tok[i] =  AMRCorpusTranslate[t]
        self.tokens = tok
        self.source['amrRaw']    = amr_raw
        self.source['amrText']   = amr_line
        self.source['metadata']  = metadata
        self.singleComponentGraph = None
        self.multiSentIX         = multiSentIX  # 0 if standalone, one-based index if multi
        

        p = re.compile(r'\s\s\s')
        m = p.search(metadata['tok'])
        if (m):
            self.parseProblem = True
        else:
            self.parseProblem = False        
            
            a = [{}] * len(self.tokens) 
            for e in alignments:
                if (e ==''):
                    continue
                ix = int(e.split("-")[0])
                treeRef = e.split("-")[1]
                if ix >= len(a):
                    print '--------------'
                    print amr_raw
                    print self.tokens
                    print tok
                    print metadata
                    print metadata['tok'].split()
                    print 'DEBUG', a
                    print 'DEBUG', e
                    print ix
                    #continue
                
                if not 'e' in a[ix]:
                    a[ix] = {'r':[], 'e':[]}
                if ('r' == e.split(".")[-1]):
                    a[ix]['r'].append(treeRef)
                else:
                    a[ix]['e'].append(treeRef)
            self.alignments = a

    def setPCFG(self, pcfg):
        self.pcfg = pcfg
    def setTokens(self, tokens):
        self.tokens = tokens
    def __repr__(self):
        from pprint import pformat
        return pformat(vars(self), indent=4, width=1)

class TXConvert:
    conversionHash = None
    conversionHashFn = 'conversionHash.pcl'
    
    @staticmethod
    def init(fn):
        TXConvert.conversionHashFn = fn
        TXConvert.conversionHash = None
        if (fn==None):
            TXConvert.conversionHash = {'noConversion': True}

    @staticmethod
    def convert(intoken):
        if not TXConvert.conversionHash:
            print 'LOADING HASH TABLE FOR CONVERSION'
            TXConvert.conversionHash = pickle.load(  open( TXConvert.conversionHashFn ) )  
        if intoken in TXConvert.conversionHash:       
            return TXConvert.conversionHash[intoken]
        else:
            return intoken


def get_amr_and_comment_line(input_f):

    """
    Read the file containing AMRs. AMRs are separated by a blank line.
    Each call of get_amr_line() returns the next available AMR (in one-line form).
    Note: this function does not verify if the AMR is valid

    """
    cur_amr = []
    raw_amr = ""
    cur_comment = []
    has_content = False
    for line in input_f:
        
        line = line.encode("utf-8").replace('\xe2\x80\x99', " '")
        line = line.encode("utf-8").replace('\xe2\x80\x9c', " ")
        line = line.encode("utf-8").replace('\xe2\x80\x9d', " ")
        line = line.encode("utf-8").replace('\xc2\xb7', " ")
        line = line.encode("utf-8").replace('\xc2\xa3', " ")
        line = line.encode("utf-8").replace('\xe2\x80\x93', " ")
        
        line = line.encode("utf-8").replace('\xe2\x80\x98', " ")
        line = line.encode("utf-8").replace('\xc3\xa7', " ")
        line = line.encode("utf-8").replace('\xe2\x80\x94', " ")
        line = line.encode("utf-8").replace('\xe2\x80\xa6', " ")
        line = line.encode("utf-8").replace('\xc3\xa9', " ")
        line = line.encode("utf-8").replace('\xc3\xa1', " ")
        
        if line.strip() == "":
            if not has_content:
                # empty lines before current AMR
                continue
            else:
                # end of current AMR
                break
        if line.strip().startswith("#") and ('::' not in line):
            continue
        elif line.strip().startswith("#"):
            has_content = True                                       
            raw_amr += line 
            cur_comment.append(line.strip())
        else:
            has_content = True
            raw_amr += line 
            cur_amr.append(line.strip())
    metadata = {}        
    for c in cur_comment:  
        cc= c[1:] # the hashmark character
        for p in cc.strip().split('::'):
            p = p.strip()
            if len(p):
                tokens = p.split(' ')
                key = tokens[0]
                val = " ".join(tokens[1:])
                metadata[key] = val
            
    return "".join(cur_amr) , metadata, raw_amr
      
#        zzsplit = re.split('(\.|\,|\)|\(|\-)', zz)             
def tokenizeSentence(snt): 
    t = []           
    zzz = []
    snt = re.sub ('\/', ' ', snt) # get rid of forward slashes, change to spaces
    for splitOnSpace in snt.split(' '):
        if not (splitOnSpace == ' '):
            for splitIncludingParens in re.split(r'(\)|\(|\")', splitOnSpace):
                if not splitIncludingParens=='':
                    zzz.append(splitIncludingParens)
    
    for zz in zzz:   
    #for zz in snt.split(' '):
        m = re.match("Mr\.", zz)
        if m:
            t.append(zz)
            continue
        m = re.match("\S\.", zz)
        if m:
            t.append(zz)
            continue
        m = re.search("Mrs\.", zz)
        if m:
            t.append(zz)
            continue
        m = re.match("(.*)(n\'t)$", zz)
        if m:
            t.append(m.group(1))
            t.append(m.group(2))
            continue
        m = re.match("(.*)(\'s)$", zz)
        if m:
            t.append(m.group(1))
            t.append(m.group(2))
            continue
        m = re.match("(.*)(\'ve)$", zz)
        if m:
            t.append(m.group(1))
            t.append(m.group(2))
            continue
        m = re.match("(.*)(\'m)$", zz)
        if m:
            t.append(m.group(1))
            t.append(m.group(2))
            continue
        m = re.match("(.*)(\'ll)$", zz)
        if m:
            t.append(m.group(1))
            t.append(m.group(2))
            continue
        m = re.match("(.*)(\'re)$", zz)
        if m:
            t.append(m.group(1))
            t.append(m.group(2))
            continue


        zzsplit = re.split(r"(\.|\,|\)|\(|\-|\;|\!|\?|\'|\:|\$|\"|\/)", zz)             
        for z in zzsplit:
            if len(z):
                if z=='-':
                    z='@-@'
                t.append(z)
                
    tokens = []             
    for z in t: 
        if len(z):
            tokens.append(z)           
    return tokens
    
 
def getHierarchy2nodeName(a):
    
    # first get rid of stuff between quotes.
    # then, assign hierarchy to node names based on :, ), and ( 
    re.sub("[\"][^\"]+[\"]",'',a)
    print a
    levelDesc = [0] * 20
    iLevel=0
    line=''
    for _,c in enumerate(a):
        if c=='(':
            print levelDesc[:8],
            print '( %d %s ' % ( iLevel, line)
            levelDesc[iLevel] += 1
            iLevel += 1
            line=''
        elif c==')':
            print levelDesc[:8],
            print ') %d %s ' % ( iLevel, line)
            iLevel -= 1
            levelDesc[iLevel] -= 1
        else:
            line += c    
            

def cleanRawMetadataTokens(tokens):  
    toks=[]
    for t in tokens.strip().split():
        t = re.sub('/', '', t)
        #t = re.sub('\)', '', t)
        #t = re.sub('\(', '', t)
        #t = re.sub('\:', '', t)
        #t = re.sub('\"', '', t)
        if (t=='') or (t==' '):
            t = '<b>'
        toks.append(t) # cannot alter the number of tokens because it screws up alignment
    return toks
   
                
def readAllAMRFromFile(fn, ignore=0):
    
    
    def getMultiRoots(p):
        roots = []
        rels = []
        for i,_ in enumerate(p.nodes):
            if p.node_values[i] == 'multi-sentence':    
                for _to, ekind in  p.relations[i].iteritems():
                    roots.append(_to)
                    rels.append(ekind)
                return roots, rels
        return roots, rels


    def getSubgraphNodesFromRoot(G, top, nList):
        for sn in G.successors(top):
            nList.append(sn)                
            getSubgraphNodesFromRoot(G, sn, nList)
        return nList
    
    
    
    infile = codecs.open(fn, encoding='utf-8')  # note: changed from latin-1 on 7/8/16
    allSentences = []
    pcount = 0
    stats = {'multiBad':0, 'multiGood':0, 'single':0 }
    
    pRejection = re.compile(r'\s\s')
    
    while True:
        amr_line, metadata, amr_raw = get_amr_and_comment_line(infile)
        # alignments file has :alignments and :tok,
        # non-aligned has :snt
        if metadata:
            if 'alignments' in metadata:
        
                m = pRejection.search(metadata['tok'])
                if (m):
                    continue # skip this sentence
                
            else:
                metadata['alignments'] = ''
            
                


        
        pcount += 1
        print 'PARSING AMR NUMBER %d from file %s' % (pcount, os.path.basename(fn)), metadata
        if ignore:
            if pcount < ignore:
                continue
        #if amr_line:  # AMR is in the description
        #    raise RuntimeError('not training code')
        elif metadata: # no amr description
            if not 'tok' in metadata:
                sents = tokenizer.tokenize(metadata['snt'])  # breaks on Gov. Pfc. and other abbrevs.
                for sentIX, sent in enumerate(sents):
                    metadata['tok'] = ' '.join(tokenizeSentence(sent))
                    parsedAMR=None
                    s = Sentence( metadata['alignments'].strip().split(" "), parsedAMR,  cleanRawMetadataTokens(metadata['tok']), 
                                  amr_raw, amr_line, metadata, multiSentIX=sentIX+1)
                    allSentences.append(s)    
            else:
                stats['single'] += 1
                parsedAMR=None
                s = Sentence( metadata['alignments'].strip().split(" "), parsedAMR,  cleanRawMetadataTokens(metadata['tok']), 
                              amr_raw, amr_line, metadata )
                allSentences.append(s)    

        else:
            break

    infile.close()
    return allSentences, stats


def readAllAMR(fn, ignore=0):
    slist = []
    stats = {'multiBad':0, 'multiGood':0, 'single':0 }
    if os.path.isdir(fn):
        for filename in sorted(os.listdir(fn)):   # critical: sort the list, it is OS dependent!
            # Should not start with dot, should end with txt
            fileOK = re.search('^[^\.].*\.txt$', filename)  
            if (fileOK):
                ns, nstats = readAllAMRFromFile('%s/%s' % (fn,filename), ignore=ignore )
                slist += ns
                stats['multiBad'] += nstats['multiBad'] 
                stats['multiGood'] += nstats['multiGood'] 
                stats['single'] += nstats['single'] 

    else:
        slist, stats = readAllAMRFromFile(fn, ignore=ignore)
    return slist, stats
    



  
def addValTokenSplitsToGraph(G):
    for lbl in G.nodes():        
        z = G.node[lbl]['value'].split('~')
        kind = z[0]
        tokref = ''
        if (len(z)>1):
            tokref = z[1]
        G.node[lbl]['valOnly'] = kind
        G.node[lbl]['tokref'] = tokref 
    return G
  




def getSentenceDFTagList():
    # txFunc is not a concept Identifier.  Keep the name for now.
    # txArg is replaced with sense, nameCategory, wiki
    # the previous incoming txRel and txRelSrc are replaced with the four nar%d_lbl and nar%d_ix pairs for OUTGOING relations
    tagList = [ 'sentIX', 'wordIX', 'words', 'txFunc', 'txBIOES', 'txBIOESProb', 'NcatProb', 'sense', 'nameCategory', 'wiki', 'top', 'kind', 'nnParent', 'nnChild'  ]
    for i in range(4): # max of four args per predicate
        tagList += [  'ar%d_arg' % i, 'ar%d_ix' % i  ]
    for i in range(4): # max of four nonarg relations per nonPred
        tagList += [  'nar%d_lbl' % i, 'nar%d_ix' % i  ]
    for i in range(4): # max of four attributes per concept
        tagList += [  'attr%d_lbl' % i, 'attr%d_val' % i  ]
    for i in range(8): # store all wordIX's
        tagList += [  'wordIX%d' % i   ]
    tagList += ['pVectorL0',  'pVectorL0Args',  'pVectorL0Nargs',  'pVectorL0Attr']    
    return tagList

def getResultsDFTagList():
    tagList = [ 'sentIX', 'wordIX', 'words', 'kind' ] + [ 'txBIOES', 'txBIOESProb', 'NcatProb', 'sense', 'NcatResult', 'WKLink' ] 
    for i in range(4): # max of four args per predicate
        tagList += [  'ar%d_arg' % i, 'ar%d_ix' % i  ]
    for i in range(4): # max of four nonarg relations per nonPred
        tagList += [  'nar%d_lbl' % i, 'nar%d_ix' % i  ]
    for i in range(4): # max of four attributes per concept
        tagList += [  'attr%d_lbl' % i, 'attr%d_val' % i  ]    
    tagList += ['pVectorL0',  'pVectorL0Args',  'pVectorL0Nargs',  'pVectorL0Attr']    
    return tagList
 



def mergeSentenceDataFrames(inFn, sTypes, outFn, sents=None, sentenceAttr='referenceDFrame'):
    '''
    merges dataframes specified by sentenceAttr='referenceDFrame' from all sentences into one dataframe
      which is returned.  Each input dataframe should have an index (sentIX) embedded so that we can determine
      the source in the larger dataframe.
    :param inFn:
    :param sTypes:
    :param outFn:
    :param sents:
    :param sentenceAttr:
    '''
    if inFn:
        sents = pickle.load(  open( inFn ) ) 
    merged = {}
    for dbType in sTypes:
        df = pd.DataFrame()
        dfClump = pd.DataFrame()
        for sentIX, sentence in enumerate(sents[dbType]):
            if not (sentIX % 100):
                df = df.append( dfClump, ignore_index=True)
                dfClump = pd.DataFrame()
                print dbType, 'clumped', sentIX
            if not hasattr(sentence, sentenceAttr):
                continue
            dfClump = dfClump.append( getattr(sentence, sentenceAttr), ignore_index=True)
        df = df.append( dfClump, ignore_index=True) # finish it off
        merged[dbType] = df
    if (outFn):
        pickle.dump( merged, open( outFn, "wb" ) )
    return merged 


