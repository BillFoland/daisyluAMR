#from daisylu_vectors import *
import networkx as nx
import pickle
import pandas as pd


from daisylu_config import *
from math import exp, log
from sentences import *
from daisylu_vectors import *
from daisylu_system import *
from pprint import pprint
from daisylu_endec import getSubGraph, getSourceDestKindForRelation



def prohibitConnectionTo(logProbs, probs, destLbl):
    for i,t in enumerate(logProbs[destLbl]):
        logProbs[destLbl][i] = -30.0                                
        probs[destLbl][i] = 1e-10                                
    logProbs[destLbl][0] = -0.0001    
    probs[destLbl][0] = 1.0 - 1e-10    


def mergeNodes(G, nodes):
    """
    Merges the selected `nodes` of the graph G into the first node,
    meaning that all the edges that pointed to or from one of these
    `nodes` will point to or from the first node.
    """
    def reverseLabels(d):
        if 'rLabel' in d:
            d['rLabel'] = d['rLabel'] + '-of'        
        if 'label' in d:
            d['label'] = d['label'] + '-of'   
        return d       
    for n1,n2,data in G.edges(data=True):
        # For all edges related to one of the nodes to merge,
        # make an edge going to or coming from the `new gene`.
        if n1 in nodes[1:]:
            source = nodes[0]; dest = n2
            G.add_edge(source, dest, data)
            allCycles = [c for c in nx.simple_cycles(G)]
            if allCycles:
                print 'cycle induced by adding ', source, dest, data
                G.remove_edge(source, dest)  
                G.add_edge(dest, source, reverseLabels(data) )
        elif n2 in nodes[1:]:
            source = n1; dest = nodes[0]
            G.add_edge(source, dest, data)
            allCycles = [c for c in nx.simple_cycles(G)]
            if allCycles:
                print 'cycle induced by adding ', source, dest, data
                G.remove_edge(source, dest)  
                G.add_edge(dest, source, reverseLabels(data) )
    
    for n in nodes[1:]: # remove the merged nodes
        G.remove_node(n)
        
def forceICorefs(sents):    
    for i,sentence in enumerate(sents['test']):
        print 'corefs', i
        G = sentence.singleComponentGraph['graph'] 
        topLbl = sentence.singleComponentGraph['topLbl']
        topIX = sentence.singleComponentGraph['topIX'] 
        gList = {}
        for lbl in G.nodes():
            v = G.node[lbl]['valOnly']
            if not v in gList:
                gList[v]=[]
            gList[v].append(lbl) 
        if ('i' in gList):
            if topLbl in gList['i']:
                print gList    
                sentence.singleComponentGraph['topLbl'] = gList['i'][0]
            if  (len(gList['i']) > 1):
                mergeNodes(G, gList['i'])        
        #showPDFFromSentenceList( [sentence], [[[G,'']]] ) #,   auxGraphs,  pandasDataFrames )        
        #sentence.singleComponentGraph['graph'] = G
    return sents

def removeQuantHMMAttrs(sents):
    for i,sentence in enumerate(sents['test']):
        print 'quant HMM removal', i
        G = sentence.singleComponentGraph['graph'] 
        for lbl in G.nodes():       
            for a in G.node[lbl]['attributes'].keys():
                if (a == 'quant') and (G.node[lbl]['attributes'][a] == 'HMM'):
                    del(G.node[lbl]['attributes'][a])
    return sents
        
def translateCountryCat(sents):
    txWiki={}
    
    txWiki["Canada_national_men's_ice_hockey_team"]           =    "Canada" 
    
    txWiki["Cuban_Revolution"]          =    "Cuba" 
    
    txWiki["Economy_of_Afghanistan"]          =    "Afghanistan" 
    
    txWiki["Flag_of_Japan"]          =    "Japan" 
    
    
    
    txWiki['History_of_Japan-Korea_relations'] = 'Japan'
    
    
    
    txWiki["Indian_Defence"]              =    "India" 
    txWiki["Israelis"]                    =    "Israel" 
    txWiki["Italians"]                    =    "Italy" 
    txWiki["Politics_of_Brazil"]          =    "Brazil" 
    txWiki["Politics_of_Kyrgyzstan"]          =    "Kyrgyzstan" 
    txWiki["Politics_of_North_Korea"]          =    "North_Korea" 
    txWiki["Politics_of_Vietnam"]          =    "Vietnam" 
    txWiki["President_of_Brazil"]          =    "Brazil" 
    
    txWiki["President_of_South_Africa"]          =    "South_Africa" 
    txWiki["Government_of_South_Africa"]          =    "South_Africa" 
    
    txWiki["Islamic_Republic_of_Iran_Broadcasting"]   =   "Iran"   
    txWiki["Islamic_republic"]                        =   "Iran" 
    txWiki["Islamic_republic"]                        =   "Iran" 
    txWiki["President_of_Iran"]                       =   "Iran" 
    txWiki["Nuclear_program_of_Iran"]                 =   "Iran" 
    txWiki["Politics_of_Iran"]                        =   "Iran" 
    
    
    txWiki["Peruvian"]          =    "Peru" 
    txWiki["Peruvian_Army"]          =    "Peru" 
    txWiki["Military_of_Peru"]          =    "Peru" 
    
    
    txWiki["France_Info"]          =    "France" 
    txWiki["French_people"]          =    "France" 
    txWiki["Government_of_France"]          =    "France" 
    
    
    txWiki["Chancellor_of_Germany"]          =    "Germany" 
    
    txWiki["Han_Chinese"]          =    "China" 
    txWiki["Culture_of_China"]          =     "China" 
    txWiki["Foreign_Minister_of_the_People's_Republic_of_China"]          =  "China"  
    txWiki["China_Investment_Promotion_Agency"]          =    "China" 
    txWiki["China_national_football_team"]          =    "China" 
    txWiki["Mainland_China"]          =    "China" 
    txWiki["People's_Republic_of_China"]          =  "China"  
    txWiki["Premier_of_the_People's_Republic_of_China"]          =  "China"   
    
    
    txWiki["Prime_Minister_of_Russia"]              =    "Russia" 
    txWiki["Russia_national_football_team"]         =      "Russia" 
    txWiki["Russian_Ministry_of_Defence"]           =      "Russia" 
    txWiki["Armed_Forces_of_the_Russian_Federation"]          =    "Russia" 
    txWiki["Government_of_Russia"]          =     "Russia" 
    
    
    txWiki["Government_of_Australia"]          =    "Australia" 
    txWiki["Prime_Minister_of_Australia"]          =    "Australia" 
    txWiki["Australia_national_rugby_union_team"]             =     "Australia" 
    txWiki["Australian_Defence_Force"]                        =     "Australia" 
    
    
    txWiki["Federal_government_of_the_United_States"] =    "United_States" 
    txWiki["History_of_the_United_States"]            =      "United_States" 
    txWiki["Military_of_the_United_States"]           =     "United_States" 
    txWiki["President_of_the_United_States"]          =    "United_States" 
    txWiki["American_Missionary_Association"]         =      "United_States" 
    txWiki["The_Pentagon"]                            =      "United_States" 
    txWiki["U.S._state"]                              =      "United_States" 
    txWiki["United_States_Department_of_State"]       =      "United_States" 
    txWiki["United_States_Marine_Corps"]              =      "United_States" 
    txWiki["United_States_dollar"]                    =      "United_States" 
    
    txWiki['United_States-Iran_relations']            = "United_States"
    txWiki["Democracy"]                               =      "United_States" 
    
    txWiki["Diplomatic_missions_of_Israel"]           =    "Israel" 
    txWiki["Politics_of_Israel"]                      =    "Israel" 


    txWiki["Chinese_language"]           =    "China" 
    txWiki["French_language"]            =    "France" 
    txWiki["German_language"]            =    "Germany" 
    txWiki["Japanese_language"]          =    "Japan" 
    txWiki["Korean_language"]            =    "Korea" 
    txWiki["Mongolian_language"]         =    "Mongolia" 
    txWiki["Nepali_language"]            =    "Nepal" 
    txWiki["Russian_language"]           =    "Russia" 
    txWiki["Somali_language"]            =    "Somalia" 
    txWiki["Spanish_language"]           =    "Spain" 
    txWiki["Ukrainian_language"]         =    "Ukraine" 
    txWiki["Vietnamese_language"]        =    "Vietnam" 

    for i,sentence in enumerate(sents['test']):
        print 'country category translation', i
        G = sentence.singleComponentGraph['graph'] 
        for lbl in G.nodes():    
            if (G.node[lbl]['valOnly']=='country'):   
                for a in G.node[lbl]['attributes'].keys():
                    if (a == 'wiki'):
                        unquoted = re.sub(r"\"", r"",  G.node[lbl]['attributes'][a] )
                        if unquoted in txWiki:
                            G.node[lbl]['attributes'][a] = '"' + txWiki[unquoted] + '"'
                            print unquoted, G.node[lbl]['attributes'][a]
    return sents


    
def addDotLabelsToProbGraph(G):
    for lbl in G.nodes():
        attrString = ''
        for a in G.node[lbl]['attributes'].keys():
            attrString += '\n%s: %s' % (a, G.node[lbl]['attributes'][a])
        nodeProb = -1.00
        if ('prob' in G.node[lbl]):
            nodeProb = G.node[lbl]['prob']

        G.node[lbl]['label'] = '%s (%s)\n%s\n%s%s' % (G.node[lbl]['valOnly'], lbl, 
                                                      '%.2f' % (nodeProb * 100.0),
                                                      G.node[lbl]['tokref'], attrString)
        if True: #(G.node[lbl]['tokref'] != ''):
            G.node[lbl]['style'] = 'filled'
            G.node[lbl]['color'] = 'black'
            if nodeProb < 0.4:
                G.node[lbl]['fillcolor'] = 'pink'
            else:
                G.node[lbl]['fillcolor'] = 'white'
    for n1,n2,_ in G.edges(data=True):
        G.edge[n1][n2]['label'] = G.edge[n1][n2]['rLabel'] + '\n%.2f' % (G.edge[n1][n2]['prob'] * 100.0)
    return G



def createSingleComponentAMRGraphFromDataFrameProbabilities(sent, features, verbose=False, 
                                                            forceSubGroupConnectionThreshold = 0.35,
                                                            conceptRejectionThreshold=0.0):
    
    return NEWcreateSingleComponentAMRGraphFromDataFrameProbabilities(sent, features, verbose=verbose, 
                                                            forceSubGroupConnectionThreshold=forceSubGroupConnectionThreshold,
                                                            conceptRejectionThreshold=conceptRejectionThreshold) 





def getAdjacencyDF(gC):
    tos = nx.topological_sort(gC)
    df = pd.DataFrame( columns=tos )
    for i in tos:
        df.loc[i]=[0] * len(tos)
    for e in gC.edges_iter(data='rLabel'):
        df.loc[e[0],e[1]] = e[2]
    return df


def normalizeLogProbs1d(inProbs):
    probs={}
    logProbs={}
    
    for wordIX in inProbs.keys():
        tagLogProbs = inProbs[wordIX]
        probsUnNormed = [exp(x) for x in tagLogProbs]
        probSum = sum(probsUnNormed)
        p = [x/probSum for x in probsUnNormed] 
        probs[wordIX]  = p 
        #print 'DEBUG',p
        logProbs[wordIX]  = [log(x+1e-20) for x in p] 
    return logProbs, probs



def getNormalizedLogProbs1D(df, column):
    # wordIX and sentIX are zero-based here
    probs={}
    logProbs={}
        
    for ix, row in df.iterrows():
        v = row[column]
        wordIX = row['wordIX']
        #print  column, wordIX, 'v, and list', v
        if pd.isnull(v):
            continue
        else:     
            lst = floatCSVToList(v) 
            #print lst
            probsUnNormed = [exp(x) for x in lst[0]]
            probSum = sum(probsUnNormed)
            p = [x/probSum for x in probsUnNormed] 
            probs[wordIX] = p 
            logProbs[wordIX] = [log(x+1e-20) for x in p] 
            
    return logProbs, probs    


def normalizeLogProbs2d(inProbs):
    probs={}
    logProbs={}
    
    for wordIX in inProbs.keys():
        for wix,tagLogProbs in inProbs[wordIX].items():
            probsUnNormed = [exp(x) for x in tagLogProbs]
            probSum = sum(probsUnNormed)
            p = [x/probSum for x in probsUnNormed] 
            if not wordIX in probs:
                probs[wordIX] = {}
                logProbs[wordIX] = {}
            probs[wordIX][wix] = p 
            logProbs[wordIX][wix] = [log(x+1e-20) for x in p] 
    return logProbs, probs


# First get the logProb array from the df,
# Then normalize the logProb array and create a prob array from it.
# The normalization can be used by itself during graph construction.
def getNormalizedLogProbs2D(df, column):
    # wordIX and sentIX are zero-based here
    probs={}
    logProbs={}
    conceptWordIX = [int(x) for x in df['wordIX'].tolist()]
    for ix, row in df.iterrows():
        v = row[column]
        if pd.isnull(v):
            continue
        else:     
            wordIX = int(row['wordIX'])
            lst = floatCSVToList(v)   
            for wix,tagLogProbs in enumerate(lst):
                if not wordIX in logProbs:
                    logProbs[wordIX] = {}
                logProbs[wordIX][wix] = tagLogProbs
    logProbs, probs = normalizeLogProbs2d(logProbs)
    return logProbs, probs    




def xxxxgetNormalizedLogProbs2D(df, column):
    # wordIX and sentIX are zero-based here
    probs={}
    logProbs={}
    conceptWordIX = [int(x) for x in df['wordIX'].tolist()]
    for ix, row in df.iterrows():
        v = row[column]
        if pd.isnull(v):
            continue
        else:     
            wordIX = int(row['wordIX'])
            lst = floatCSVToList(v) 
            for wix,tagLogProbs in enumerate(lst):
                probsUnNormed = [exp(x) for x in tagLogProbs]
                probSum = sum(probsUnNormed)
                p = [x/probSum for x in probsUnNormed] 
                if not wordIX in probs:
                    probs[wordIX] = {}
                    logProbs[wordIX] = {}
                probs[wordIX][wix] = p 
                logProbs[wordIX][wix] = [log(x) for x in p] 
            
    return logProbs, probs    

def getMostProbableTOP(df, dbf):
    # one concept is the most likely top
    # normalize probabilities on a per tag basis
    # then select the most probable concept to assign as TOP from the result
    dbfn = getSystemPath('daisylu') + 'data/%s' % dbf
    _, features, _ = readAMRVectorDatabase(dbfn)
    topIX = features['attr']['t2i']['TOP']
    print df, features['attr']['t2i']
    logProbs, probs = getNormalizedLogProbs1D(df, 'pVectorL0Attr' )
    pprint(logProbs)    
    pprint(probs) 
    maxP = -1
    maxIX = -1
    for wordIX in probs.keys():
        p = probs[wordIX][topIX]
        if p > maxP:
            maxP = p
            maxIX = wordIX
    print ' max index is ', maxIX, 'max probability is ', maxP    
    return maxIX, maxP
        
        
def recursiveNodeDescription(gC, n, tos, level):
    # New 1/8/16:
    # This is a fully connected diGraph, but we need to source it from the given n node.  To do this,
    # edges will need to be reversed.
    
    
    
    # Determine edges which need to be flipped in order to describe directed graph
    # Traverse tree, starting from top item left in topological sort list
    # As each node is described, remove it from the topoSort list.
    # After this is done, if there are remaining nodes, an edge will need to be reversed 
    if n in tos:
        tos.remove(n)
    else:
        return '%s' % n
    nValue = gC.node[n]['value']
    nAttr  = gC.node[n]['attributes']
    if r'/' in nValue:
        nValue = '"' + nValue + '"'
    s =  '(%s / %s ' % (n, nValue)
    for a in nAttr.keys():
        if a != 'TOP':                            # the TOP attribute is not output to txt, it is implied 
            s += ':%s %s ' % ( a, nAttr[a])
                        
    eList = gC.out_edges(n, data='rLabel')
    
    
    for i,e in enumerate(eList):
        if (len(e[2])>6) and (e[2][-6:]=='-of-of'):   
            a = list(e)
            a[2] = a[2][:-6]
            eList[i] = tuple(a)

    def wordIX_compare(x, y):
        matchObj = re.match( r'[^\.]*\.(\d+)', x[1])
        xix = int(matchObj.group(1)) % 1000
        matchObj = re.match( r'[^\.]*\.(\d+)', y[1])
        yix = int(matchObj.group(1)) % 1000
        return xix - yix
    
    # sorting ops, decreases score by 0.6%.
    # why? too late to investigate now (1/30/16)
    sortOps=False
    if (sortOps):
        eList.sort(cmp=wordIX_compare)

    opIX = 1    
    for e in eList:
        argLabel = e[2]
            
        if (sortOps):
            matchObj = re.match( r'op\d', argLabel)
            if matchObj:
                print '     Changing %s to ' % argLabel,
                argLabel = 'op%d' % opIX
                print argLabel
                opIX += 1
        
        level += 1
        s+= '\n'
        s+= '\t' * level 
        s+= ':%s ' % argLabel + recursiveNodeDescription(gC, e[1], tos, level)
        level -= 1
    return s + ' )'

def reverseEdgesIfNeeded(gC, sent=None):  
    ag = [ (gC, 'start') ] 
    tos = nx.topological_sort(gC)
    while True:
        print '.'
        ag.append( (gC, 'top of reverse loop') ) 
        gC = removeCycles(gC)

        allCycles = [c for c in nx.simple_cycles(gC)]
        if allCycles:
            print 'punting.............'
            exit(1)

        #print 'tos is ', tos
        n = tos[0]
        nList = list(nx.dfs_preorder_nodes( gC , n ))
        #print 'traversal from %s is ' % n, nList
        if (len(nList) == len(tos)):
            ag.append( (gC, 'reverse if needed Done') ) 
            break  # all nodes covered by traversal               
        df = getAdjacencyDF(gC)
        #print 'full df is \n', df
        df = df.drop(nList)
        df = df[nList]
        #print df
        # one of the edges in df needs to be reversed to include weakly connected subgraphs, for now, find the first one topologically
        for n in tos:
            if n in nList:
                fromList = df[ df[n] != 0 ].index.values.tolist()
                if fromList:
                    oldLabel = gC.edge[fromList[0]][n]['rLabel']
                    if oldLabel[-3:]=='-of':
                        newLabel = oldLabel[:-3]
                    else:
                        newLabel = oldLabel + '-of'
                    gC.remove_edge(fromList[0], n)
                    gC.add_edge(n, fromList[0], label=newLabel)
                    break
    return gC, ag   
        
def joinByReversingEdge(G, tos, notConnected):
    for n in notConnected:
        for dest in tos:
            if G.has_edge(n,dest):
                print 'reversing connection:',   n, dest, 
                print G.edge[n][dest]
                oldLabel = G.edge[n][dest]['rLabel']
                oldProb  = G.edge[n][dest]['rLabel']
                print  oldLabel 
                newLabel = oldLabel + '-of'
                G.remove_edge(n, dest)
                G.add_edge(dest, n, rLabel=newLabel)
                allCycles = [c for c in nx.simple_cycles(G)]
                if (allCycles):
                    G.remove_edge(dest, n)
                    G.add_edge(n, dest, rLabel=oldLabel)
                else:
                    return G
    print 'WARNING: SHOULD NEVER GET HERE <----------------------------------------------'            


def adjustProbabilities(G, ixToLbl, conceptWordIX, tagIX, destIX, sourceIX, rType, rProb, edgeLabel):

    sourceLbl = ixToLbl[sourceIX]
    destLbl   = ixToLbl[destIX]
    
    

    if rType=='args':
        logProbs   = G.node[sourceLbl]['argLogProbs']
        # no other connections of this arg type will be allowed
        for c in conceptWordIX:
            if c in logProbs:
                logProbs[c][tagIX] = -30.0                
        G.node[sourceLbl]['argLogProbs'], G.node[sourceLbl]['argProbs'] = normalizeLogProbs1d(logProbs)
        
        prohibitConnectionTo(G.node[sourceLbl]['argLogProbs'], G.node[sourceLbl]['argProbs'],  destIX)             # mark 'O' as the choice so that future passes will ignore
        prohibitConnectionTo(G.node[sourceLbl]['nargLogProbs'], G.node[sourceLbl]['nargProbs'],  destIX)      

        # prohibit feedback connections from the newly connected node.
        logProbs   = G.node[destLbl]['argLogProbs']
        if logProbs:
            prohibitConnectionTo(G.node[destLbl]['argLogProbs'], G.node[destLbl]['argProbs'],  sourceIX)      
        
        # make other narg connections from the destination impossible
        logProbs   = G.node[destLbl]['nargLogProbs']
        if logProbs:
            prohibitConnectionTo(G.node[destLbl]['nargLogProbs'], G.node[destLbl]['nargProbs'],  sourceIX)      
        
    else:
        # make other narg connections to the destination impossible
        prohibitConnectionTo(G.node[sourceLbl]['nargLogProbs'], G.node[sourceLbl]['nargProbs'],  destIX)      

        # make other arg connections to the destination impossible
        logProbs   = G.node[sourceLbl]['argLogProbs']
        if logProbs:
            prohibitConnectionTo(G.node[sourceLbl]['argLogProbs'], G.node[sourceLbl]['argProbs'],  destIX)      

        # prohibit feedback connections from the newly connected node.
        logProbs   = G.node[destLbl]['argLogProbs']
        if logProbs:
            prohibitConnectionTo(G.node[destLbl]['argLogProbs'], G.node[destLbl]['argProbs'],  sourceIX)      
         
        # make other narg connections from the destination impossible
        logProbs   = G.node[destLbl]['nargLogProbs']
        if logProbs:
            prohibitConnectionTo(G.node[destLbl]['nargLogProbs'], G.node[destLbl]['nargProbs'],  sourceIX)      
        
    return

 
def findMostProbableConnection(G, ixToLbl, conceptWordIX ):
    # Find most probable connection by examining the node probability tables
    candidates = {}
    for sourceWordIX in conceptWordIX:      

        sourceLbl = ixToLbl[sourceWordIX]
             
        argProbs      = G.node[sourceLbl]['argProbs']      
        nargProbs     = G.node[sourceLbl]['nargProbs']
        if argProbs:  
            for destWordIX,lst in argProbs.items():
                if destWordIX==sourceWordIX:
                    continue
                for ti,p in enumerate(lst):
                    if ti==0:
                        continue 
                    candidates[p] = [ ti, destWordIX, sourceWordIX, 'args']
        if nargProbs:  
            for destWordIX,lst in nargProbs.items():
                if destWordIX==sourceWordIX:
                    continue
                for ti,p in enumerate(lst):
                    if ti==0:
                        continue
                    candidates[p] = [ ti, destWordIX, sourceWordIX, 'nargs']
                        
    orderedProbs = collections.OrderedDict(sorted(candidates.items(), reverse=True))
    return  orderedProbs



def compressSpans(df, verbose=False):
    """
    Create a results data frame with exact copies from the target dataframe, used to test
    graph reconstruction, for example.
    """
    def groups(v):    
        istack=[]
        rcount = len(v.index)
        for ri in range(rcount):
            BIOES = v.iloc[ri]['txBIOES'][0]
            if BIOES == 'O':
                istack = []
            elif BIOES == 'S':
                istack = [ri]
                yield istack
            elif BIOES == 'B':
                istack = [ri]
            elif BIOES == 'I':
                istack.append(ri)
            elif BIOES == 'E':
                istack.append(ri)
                yield istack        
    if (True):
        pd.set_option('display.width',   10000)
        pd.set_option('display.max_rows', 200)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.max_colwidth', 200)
        if verbose:
            print 'predicted:'
            print df
 
    
    tagList = getResultsDFTagList()
    tagList += ['allWords', 'allWordIX']
    predicted = pd.DataFrame( columns=tagList )
    level0 = df[df['wordIX']<1000]
    tokens = level0['words'].tolist()
    destinationIXtranslationPhase1={-1:-1} # re-route to category
    destinationIXtranslationPhase2={-1:-1} # route category to name node
    
    #for i,group in enumerate(groups(df)):
    #    print i, group

    df['allWordIX'] = None
    df['allWords'] = None
    for _, group in enumerate(groups(df)):
        df.set_value(group[0], 'allWordIX',  group ) 
        allWords=[]
        for ix in group:
            allWords.append( df.loc[ix,'words'] )
        df.set_value(group[0], 'allWords',  allWords ) 

    predicted = df[ ~pd.isnull(df['allWords']) ][tagList]
    
    # translate all destination indices using the dictionary (translates the one we want too)
    # for destIX in [ 'ar0_ix', 'ar1_ix', 'ar2_ix', 'ar3_ix', 'nar0_ix', 'nar1_ix', 'nar2_ix', 'nar3_ix' ]:   
    #    predicted[destIX].replace(destinationIXtranslationPhase1, inplace=True)   # arcs that used to point to the words point to the category now
    #    predicted[destIX].replace(destinationIXtranslationPhase2, inplace=True)   # category now points to the name

    if (verbose):
        pd.set_option('display.width',   10000)
        pd.set_option('display.max_rows', 200)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.max_colwidth', 200)
        print 'predicted:'
        print predicted[tagList]
    return predicted

def NEWcreateSingleComponentAMRGraphFromDataFrameProbabilities(sent, features, verbose=False, 
                                                            forceSubGroupConnectionThreshold = 0.35,
                                                            conceptRejectionThreshold=0.0):
   
    # This is the NEW function that changes, the redirection to this should be added in the normal chain
    # based on predicted dframe having the special column name ""
    # First, compress the spans, removing 'O' and compressing multiword spans to one "subgraph", with added field of "words"
    # then, connect edges.
    # finally, expand the subgraphs into single concepts, using "txFunc" and "words", connecting edges appropriately.
    

    

    predicted = sent.predictedDFrame
    expanded  = compressSpans(predicted, verbose=verbose)
    
    # this will try to create kind from words, sense, txFunc for non-txNamed
    # predictedKinds         = predictConceptKinds(expanded, None) 
    
    G = nx.DiGraph()
    ixToLbl = {}
 
    # This is where concepts can be rejected based on low probability
    conceptWordIX = []
    wixList = expanded['wordIX'].tolist()
    probList = expanded['txBIOESProb'].tolist()
    kindList = expanded['kind'].tolist()
    for i,wix in enumerate(wixList):
        if (kindList[i]=='name') or (wix > 999) or (probList[i] >= conceptRejectionThreshold):
            conceptWordIX.append( int(wix)  )
    
    conceptWordIX = expanded['wordIX'].tolist()
    
    
   
    logProbs,  _ =  getNormalizedLogProbs2D(expanded, 'pVectorL0Args')
    argLogProbs = {}
    argProbs    = {}
    for six in logProbs.keys():
        for cix in logProbs[six].keys():
            if not cix in conceptWordIX:
                for i,t in enumerate(logProbs[six][cix]):
                    logProbs[six][cix][i] = -30.0                             # make the true choice impossible for all, including to other concepts   
                logProbs[six][cix][0] = -0.0001                               # mark 'O' as the choice so that future passes will ignore
        argLogProbs[six],  argProbs[six] = normalizeLogProbs1d(logProbs[six])
        
    logProbs,  _ =  getNormalizedLogProbs2D(expanded, 'pVectorL0Nargs')
    nargLogProbs = {}
    nargProbs    = {}
    for six in logProbs.keys():
        for cix in logProbs[six].keys():
            if not cix in conceptWordIX:
                for i,t in enumerate(logProbs[six][cix]):
                    logProbs[six][cix][i] = -30.0                             # make the true choice impossible for all, including to other concepts   
                logProbs[six][cix][0] = -0.0001          # mark 'O' as the choice so that future passes will ignore
        nargLogProbs[six],  nargProbs[six] = normalizeLogProbs1d(logProbs[six])
        
        
    # create graph G.
    # First, add the concepts with attributes, keeping track of ixToNode translations
    # Next add the relations between concepts
    for ix, row in expanded.iterrows():
        wordIX = int(row['wordIX'])
        lbl = 'N%d' % wordIX
        ixToLbl[wordIX] = lbl
        G.add_node(lbl)    
        G.node[lbl]['value']         = row['txBIOES']
        G.node[lbl]['txBIOES']       = row['txBIOES']
        G.node[lbl]['allWords']      = row['allWords']
        G.node[lbl]['allWordIX']     = row['allWordIX']
        G.node[lbl]['WKLink']        = row['WKLink']
        G.node[lbl]['NcatResult']    = row['NcatResult']
        if wordIX in argLogProbs:
            G.node[lbl]['argLogProbs']   = argLogProbs[wordIX]
            G.node[lbl]['argProbs']      = argProbs[wordIX]
        else:
            G.node[lbl]['argLogProbs']   = None
            G.node[lbl]['argProbs']      = None
        if wordIX in nargLogProbs:
            G.node[lbl]['nargLogProbs']  = nargLogProbs[wordIX]
            G.node[lbl]['nargProbs']     = nargProbs[wordIX]
        else:
            G.node[lbl]['nargLogProbs']  = None
            G.node[lbl]['nargProbs']     = None
            
        if 'txBIOESProb' in row:
            G.node[lbl]['prob'] = row['txBIOESProb']
            if 'NcatProb' in row:
                G.node[lbl]['NcatProb'] = row['NcatProb']
        else:    
            G.node[lbl]['prob'] = -1

        G.node[lbl]['attributes'] = {}
        for i in range(4):
            if (not pd.isnull(row['attr%d_lbl' % i])):
                G.node[lbl]['attributes'][row['attr%d_lbl' % i]] =  row['attr%d_val' % i] 



    wordTags  = predicted['words'].tolist()
    wordIXs   = predicted['wordIX'].tolist()
    #conceptWordIX = expanded['wordIX'].tolist()
    conceptTags = []
    for i,wix in enumerate(wordIXs):
        if wix in conceptWordIX: 
            conceptTags.append(wordTags[i] + ' (N%d)' % wix)
        else:
            conceptTags.append(wordTags[i])

    maxSourceIX=1
    print 'DEBUG PLOT'
    
    
 
    # First, connect the named wiki elements together
    for ix in ixToLbl.keys():
        if (ix > 999) and (ix < 2000):  # synthesised words are given out of range word indices
            sourceLbl = ixToLbl[ix]
            destLbl   = ixToLbl[ix-1000]
            G.add_edge(sourceLbl, destLbl, { 'rLabel': 'name', 'prob': 1.0 } )   
            ixToLbl[ix-1000] = ixToLbl[ix]
            G.node[sourceLbl]['prob'] = G.node[destLbl]['NcatProb'] 

    # adjustProbabilitiesAndAddEdge(G, ixToLbl, conceptWordIX, maxTagIX, maxDestIX, maxSourceIX, maxType, features[maxType]['tokens'][maxTagIX])       
     
    graphList = []
    subGraphList = list(nx.weakly_connected_component_subgraphs(G))  # WRF TIME
    edgesAdded = 0
    maxEdgesToAdd = 800 #len(G.nodes())*3
    while len(subGraphList)>1 and (edgesAdded < maxEdgesToAdd):
        orderedProbs = findMostProbableConnection(G, ixToLbl, conceptWordIX ) # WRF TIME 
        opList = list(orderedProbs.items())
        #maxP, maxTagIX, maxDestIX, maxSourceIX, maxType
        #for k, v in orderedProbs.iteritems(): 
        #    print k, v
  
        if len(opList)==0:
            # none of the nodes contain possible connections.
            # This is the result of fragments, for example: United States, China.
            # So, add an "and" node and link everybody together to close it out.
            lbl = 'N900'
            G.add_node(lbl)    
            G.node[lbl]['value']         = 'and'
            G.node[lbl]['txBIOES']       = 'S_txAnd'
            G.node[lbl]['allWords']      = []
            G.node[lbl]['allWordIX']      = []
            G.node[lbl]['attributes']    = {}
            
            for ix in ixToLbl.keys():
                if (ix < 1000):
                    sourceLbl = 'N900'
                    destLbl   = ixToLbl[ix]
                    G.add_edge(sourceLbl, destLbl, { 'rLabel': 'op', 'prob':1.0 } )   
            subGraphList = list(nx.weakly_connected_component_subgraphs(G))
            continue
        (maxP, (maxTagIX, maxDestIX, maxSourceIX, maxType)) = opList[0]
        #print 'DEBUG, BIASED TO SELECT ARGS FIRST'
        #for (maxP, (maxTagIX, maxDestIX, maxSourceIX, maxType)) in opList:
        #    if maxType == 'args' and maxP >= forceSubGroupConnectionThreshold:
        #        break
        #    if maxP < forceSubGroupConnectionThreshold:
        #        (maxP, (maxTagIX, maxDestIX, maxSourceIX, maxType)) = opList[0]
        #        break
        
        if (maxP < forceSubGroupConnectionThreshold): # force connection between subgroups
            print 'connecting subgroups'
            subGraphList = list(nx.weakly_connected_component_subgraphs(G))  # don't recalculate unless we are getting close to full connection

            nodeToGroup = {}
            for isg, sg in enumerate(list(nx.weakly_connected_component_subgraphs(G))):
                for n in list(sg):
                    nodeToGroup[n]=isg
            for (maxP, (maxTagIX, maxDestIX, maxSourceIX, maxType)) in orderedProbs.iteritems(): 
                if nodeToGroup[ixToLbl[maxSourceIX]] != nodeToGroup[ixToLbl[maxDestIX]]:
                    break
        
        sLbl = ixToLbl[maxSourceIX]
        dLbl = ixToLbl[maxDestIX]
        rLbl = features[maxType]['tokens'][maxTagIX]
        title = 'P=%.2f: %s  (%s,%s) to (%s,%s)' % ( maxP, rLbl, G.node[sLbl]['value'], sLbl,  G.node[dLbl]['value'], dLbl)

        #print 'DEBUG PLOT'
        #plotArgNargProbMatricesForNode(G.node[ixToLbl[maxSourceIX]], features, conceptTags, title = title + ', Before Adjust')

        sourceLbl = ixToLbl[maxSourceIX]
        destLbl   = ixToLbl[maxDestIX]
        G.add_edge(sourceLbl, destLbl, { 'rLabel': features[maxType]['tokens'][maxTagIX], 'prob': maxP } )   
        allCycles = [c for c in nx.simple_cycles(G)]
        if allCycles:
            G.remove_edge(sourceLbl, destLbl)   
            print 'allCycles: ', allCycles
        else:
            edgesAdded += 1           
        adjustProbabilities(G, ixToLbl, conceptWordIX, maxTagIX, maxDestIX, maxSourceIX, maxType, maxP, features[maxType]['tokens'][maxTagIX])  
        if True: #(len(ixToLbl.keys())-edgesAdded) > 3:
            subGraphList = list(nx.weakly_connected_component_subgraphs(G))  # This has to be here, could not relocate without error
        print i, title, len(subGraphList), len(ixToLbl.keys()), edgesAdded
        
        #if len(subGraphList) < 3:
        #    GCopy = G.copy()
        #    addValTokenSplitsToGraph(GCopy)
        #    graphList.append([GCopy, title])

        
        
        #plotArgNargProbMatricesForNode(G.node[ixToLbl[maxSourceIX]], features, conceptTags, title = title + ', After Adjust')
        #plt.show()
                    
    
    topFeatureIX = features['attr']['t2i']['TOP']
    logProbs, probs = getNormalizedLogProbs1D(expanded, 'pVectorL0Attr' )
    maxP = -1
    topIX = -1
    for wordIX in probs.keys():
        p = probs[wordIX][topFeatureIX]
        if p > maxP:
            maxP = p
            topIX = wordIX
    print ' max index is ', topIX, 'max probability is ', maxP    
                
    for lbl in G.nodes():
        G.node[lbl]['attributes'].pop("TOP", None)
        if lbl == ixToLbl[topIX]:
            G.node[lbl]['attributes']["TOP"] = G.node[lbl]['value']
    
    
    ##########################################################################################
    ##########################################################################################
    ###################              Expand subgraphs here.              #####################
    ##########################################################################################
    ##########################################################################################
    unexpandedLabels = []
    for lbl in G.nodes():
        unexpandedLabels.append(lbl)
    for lbl in unexpandedLabels:
        print lbl
        txBIOES   = G.node[lbl]['txBIOES']   
        words     = G.node[lbl]['allWords']    
        allWordIX = G.node[lbl]['allWordIX']   
        txFunc = txBIOES.split('_')[1]
 
        sg = getSubGraph(txFunc, words) 
        if not sg['kind_p']:
            print 'daisylu_output.py: Error txFunc, words, sg[kind_p] ', txFunc, words, sg['kind_p']
            exit(1)
            
            
        if txFunc == 'txNamed':
            G.node[lbl]['value'] = sg['kind_p'] #G.node[lbl]['NcatResult']  
            
            link = G.node[lbl]['WKLink']
            if not link:
                link = '-' 
            G.node[lbl]['attributes']['wiki'] = '"' + link + '"' 
        else:
            G.node[lbl]['value'] = sg['kind_p']  
            for k,v in sg['attrDict_p'].iteritems():
                G.node[lbl]['attributes'][k] = v

        if sg['kind_c']:
            childLbl = lbl+'_c'
            G.add_node(childLbl) 
            G.node[childLbl]['value']         = sg['kind_c']
            G.node[childLbl]['attributes']    = {}
            G.node[childLbl]['txBIOES']  = G.node[lbl]['txBIOES']   
            G.node[lbl]['allWords']           = []
            G.node[lbl]['allWordIX']          = []


            for k,v in sg['attrDict_c'].iteritems():
                G.node[childLbl]['attributes'][k] = v
            
            # Now adjust incoming and outgoing edges from lbl to point to childLbl if appropriate.
            # Outgoing:
            """
                for n1,n2,data in G.edges(data=True):
        # For all edges related to one of the nodes to merge,
        # make an edge going to or coming from the `new gene`.
        if n1 in nodes[1:]:
            source = nodes[0]; dest = n2
            G.add_edge(source, dest, data)
            allCycles = [c for c in nx.simple_cycles(G)]
            if allCycles:
                print 'cycle induced by adding ', source, dest, data
                G.remove_edge(source, dest)  
                G.add_edge(dest, source, reverseLabels(data) )
        elif n2 in nodes[1:]:
            source = n1; dest = nodes[0]
            G.add_edge(source, dest, data)
            allCycles = [c for c in nx.simple_cycles(G)]
            if allCycles:
                print 'cycle induced by adding ', source, dest, data
                G.remove_edge(source, dest)  
                G.add_edge(dest, source, reverseLabels(data) )
            """
            
            for e in G.out_edges([lbl], data=True):
                destLbl    = e[1]
                if not 'txBIOES' in  G.node[destLbl]:
                    print 'error' , e
                destTxFunc = G.node[destLbl]['txBIOES'].split('_')[1]
                rel = G.get_edge_data(e[0], e[1])['rLabel']
                sourceKind, destKind = getSourceDestKindForRelation(txFunc, destTxFunc, rel)
                if sourceKind=='child':
                    G.remove_edge(lbl, destLbl)  
                    G.add_edge(childLbl, destLbl, e[2])
            for e in G.in_edges([lbl], data=True):
                srcLbl    = e[0]
                if not 'txBIOES' in  G.node[srcLbl]:
                    print 'error', e 
                srcTxFunc = G.node[srcLbl]['txBIOES'].split('_')[1]
                rel = G.get_edge_data(e[0], e[1])['rLabel']
                sourceKind, destKind = getSourceDestKindForRelation(srcTxFunc, txFunc, rel)
                if destKind=='child':
                    G.remove_edge(srcLbl, lbl)  
                    G.add_edge(srcLbl, childLbl, e[2])
                
            
            G.add_edge(lbl, childLbl, { 'rLabel': sg['edge_p_c'], 'prob': 1.0 } )   
           
        print
        print txFunc, words
        print sg
        print
    
    
    if len(G.nodes())<1: # empty graph
        lbl = 'N0'
        G.add_node(lbl)    
        G.node[lbl]['value']         = 'empty'
        G.node[lbl]['attributes']    = {}
        ixToLbl[0]='N0'
        topIX=0
    addValTokenSplitsToGraph(G)
    addDotLabelsToProbGraph(G)  
    return G, graphList, topIX, ixToLbl[topIX]

def getTextAMRDescription(sentence, multiSentIX, features, forceSubGroupConnectionThreshold = 0.35, conceptRejectionThreshold=0.0):

    #print pprint(sentence)
    
    #df = sentence.predictedDFrame 
    #print df
    #expanded = expandNamedSubgraphs(df, verbose=False)
    # Ensure that the most probable TOP attribute is assigned
    

    if not sentence.singleComponentGraph:
        G, aGraphs, topIX, topLbl = NEWcreateSingleComponentAMRGraphFromDataFrameProbabilities(sentence, 
                                                                                            features, 
                                                                                            forceSubGroupConnectionThreshold = forceSubGroupConnectionThreshold,
                                                                                            conceptRejectionThreshold=conceptRejectionThreshold)      
        sentence.singleComponentGraph = {}
        sentence.singleComponentGraph['graph'] = G
        sentence.singleComponentGraph['auxGraphs'] = aGraphs
        sentence.singleComponentGraph['topIX'] = topIX
        sentence.singleComponentGraph['topLbl'] = topLbl
    else:
        G = sentence.singleComponentGraph['graph'] 
        topLbl = sentence.singleComponentGraph['topLbl']
        topIX = sentence.singleComponentGraph['topIX'] 
        
    header = '# ::id %s\n' % sentence.source['metadata']['id'].strip()
    if 'snt' in sentence.source['metadata']:
        header += '# ::snt %s\n' % sentence.source['metadata']['snt'].strip()  
    header += '# ::tok %s\n' % ' '.join(sentence.tokens).strip()
    amr = ''

    #print G.nodes()
    ag = [ (G.copy(), 'predicted') ] 

    #G = removeCycles(G)
    #ag.append( (G, 'cycles corrected') )
    #gCList = list(nx.weakly_connected_component_subgraphs(G))

    while(True):
        ag.append( (G.copy(), 'changing edges') )
        allCycles = [c for c in nx.simple_cycles(G)]
        if (allCycles):
            G = removeCycles(G)   # 7/6/16 NEW
            
        tos = nx.topological_sort(G) 
        des = nx.descendants(G, topLbl)
        topSet = set(des)
        topSet.add(topLbl)
        notConnected = set(tos) - topSet
        #print 'TOP is ', topLbl
        #print 'tos is ', tos
        #print 'des is ', des
        #print 'notConnected is ', notConnected
        if not notConnected:
            break
        # one node in notConnected connects to the topSet.  Reverse it's edge.
        joinByReversingEdge(G, topSet, notConnected)
    
    
    def mapping(x):
        return 'n%d.%s' % (multiSentIX,x[1:])
    # rename all nodes with the n#.# theme 
    GRelabelled = nx.relabel_nodes(G, mapping)
    
    topNodeName = mapping( topLbl)
    tos = nx.topological_sort(GRelabelled )    
    #tos = nx.topological_sort(GRelabelled, [topIX])    
    #print 'tos is ', tos
    #print 'des is ', nx.descendants(GRelabelled, topNodeName)
    #print 'topNode is ', topNodeName
    amr = recursiveNodeDescription(GRelabelled, topNodeName, tos, 0)
    print amr
            
    return header, amr, ag        



def removeCycles(G):
    GCopy = G.copy()
    allCycles = [c for c in nx.simple_cycles(GCopy)]
    while allCycles:
        if len(allCycles[0])<2:
            GCopy.remove_edge(allCycles[0][0], allCycles[0][0]) # bizarre self-loop
        else:    
            GCopy.remove_edge(allCycles[0][0], allCycles[0][1])
        allCycles = [c for c in nx.simple_cycles(GCopy)]
    return GCopy

def stringFromSentenceStack(stack):
    s = ''
    msIXSet = set([])
    if stack:
        if len(stack)>1:
            s = stack[0][0] # header
            s+='(m / multi-sentence\n'
            for item in stack:
                (header, amr, msIX) = item
                while msIX in msIXSet:
                    msIX += 1
                    print 'Duplicate detected', msIX, msIXSet, header
                msIXSet.add(msIX)
                print msIXSet
                if (amr):
                    s += '\t' + ':snt%d ' % msIX
                    for line in amr.split('\n'):
                        s += '\t' + line + '\n'
            s+= ' )\n\n' 
        else:
            item = stack[0]
            (header, amr, msIX) = item
            if amr:                
                s = header + amr + '\n\n'
            else:
                s = header + '(MyNullSentence / X)' + '\n\n'
    return s, []




def createOutputTextFile(sents, outFn, modelInfo=None, forceSubGroupConnectionThreshold=0.35, conceptRejectionThreshold=0.0):
    tfile = open(outFn, 'wb')
    tfile.write( '# Daisylu end to end output 8/8/16\n\n' )

    if  modelInfo and 'AMRL0Args' in modelInfo:
        argsDB  =  getSystemPath('daisylu') + 'data/%s' % modelInfo['AMRL0Args']['db']
    else:
        print '\n\n WARNING SETTING DEFAULT MODELINFO \n\n'
        argsDB  = getSystemPath('daisylu') + 'data/%s' % 'AMRL0Args_990.db'
        
    if  modelInfo and 'AMRL0Args' in modelInfo:
        nargsDB =  getSystemPath('daisylu') + 'data/%s' % modelInfo['AMRL0Nargs']['db']
    else:
        print '\n\n WARNING SETTING DEFAULT MODELINFO \n\n'
        nargsDB =  getSystemPath('daisylu') + 'data/%s' % 'AMRL0Nargs_990.db'
        
    if  modelInfo and 'AMRL0Args' in modelInfo:
        attrsDB =  getSystemPath('daisylu') + 'data/%s' % modelInfo['AMRL0Attr']['db']
    else:
        print '\n\n WARNING SETTING DEFAULT MODELINFO \n\n'
        attrsDB =  getSystemPath('daisylu') + 'data/%s' % 'AMRL0Attr_990.db'
        
        
    features = {}
    _, f, _ = readAMRVectorDatabase(argsDB)
    features['args'] = f['args'] 
    _, f, _ = readAMRVectorDatabase(nargsDB)
    features['nargs'] = f['nargs'] 
    _, f, _ = readAMRVectorDatabase(attrsDB)
    features['attr'] = f['attr'] 

    
    
    sentences=[]
    auxGraphs=[]
    stack = []
    msIXSet = set([])
    lastID = None
    for sentIX in range(len(sents['test'])):
        print '\n\nsentence %d of %d\n\n' % (sentIX, len(sents['test']))
        sentence = sents['test'][sentIX]    
        currentID = sentence.source['metadata']['id']
        
        #G, aGraphs = daisylu_dryrun.createSingleComponentAMRGraphFromDataFrameProbabilities(sentence, features)
        G=None
        
        
        #df = sentence.predictedDFrame 
        #expanded = expandNamedSubgraphs(df, verbose=False)
        #G = createAMRFromDataFrame(expanded)     # returns mapping
        ag = [ (G, 'original') ]             
        if (lastID != None) and (currentID != lastID):
            s, stack = stringFromSentenceStack(stack)
            msIXSet = set([])
            tfile.write( s )
        lastID = currentID
        while sentence.multiSentIX in msIXSet:
            sentence.multiSentIX += 1
        msIXSet.add(sentence.multiSentIX)

        header, amr, gpairs = getTextAMRDescription(sentence, sentence.multiSentIX, features, 
                                                    forceSubGroupConnectionThreshold = forceSubGroupConnectionThreshold,
                                                    conceptRejectionThreshold=conceptRejectionThreshold)
        stack.append( (header, amr, sentence.multiSentIX) )
        ag.extend(gpairs)
        auxGraphs.append(ag)     
        sentences.append(sentence)
    
    s, stack = stringFromSentenceStack(stack)
    tfile.write( s )
    tfile.close()
    return s
    #showPDFFromSentenceList( sentences, auxGraphs) #,   auxGraphs,  pandasDataFrames )        



def plot_confusion_matrix(df,
                          ax=plt.gcf,
                          normalize=False,
                          tt='Confusion matrix',
                          cmap=plt.cm.Blues, textSize=5):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    cm is an array
    classes are the labels, in order
    
                name entailment neutral contradiction
0     entailment      480.0    65.0          52.0
1        neutral       59.0   334.0         104.0
2  contradiction       35.0    49.0         511.0


   contradiction  neutral  entailment
0          433.0     71.0        73.0
1           64.0    350.0        92.0
2           21.0     56.0       504.0


    """
    if len(df) > 12:
        df = df.iloc[range(10), range(11)]
    c = df.as_matrix()
    classes = c[:,0]
    classes = [cls[0] for cls in classes]
    cm = c[:,1:].astype(np.float)
    cmNormed = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    

    
    ax.imshow(cmNormed, interpolation='nearest', cmap=cmap)
    ax.set_title(tt, size=30)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks )
    ax.set_xticklabels( classes )
    ax.set_yticks(tick_marks )
    ax.set_yticklabels( classes )
    ax.grid(False)

    if normalize:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, '%.2f' % cmNormed[i, j],
                     horizontalalignment="center",
                     color="white" if cmNormed[i, j] > 0.5 else "black", size=textSize)
    else:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, '%.0f' % cm[i, j],
                     horizontalalignment="center",
                     color="white" if cmNormed[i, j] > 0.5 else "black", size=textSize)


 
    #plt.tight_layout()
    ax.set_ylabel('Ref Label', size=20)
    ax.set_xlabel('Sys Label', size=20)


if __name__ == '__main__':
    
    exit(1)
        
        