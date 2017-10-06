
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import argparse
import ConfigParser
import sys
import shlex
from string import Template


import time


import argparse
import distutils.util

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

parser.add_argument('-n',              dest='network',        default='SG',   help='SG, Args, Nargs, Attr, Cat' )
parser.add_argument('-v',              dest='vectorFn',       required = True,   help=' ')
parser.add_argument('-m',              dest='modelFn',        required = True,   help=' ')
parser.add_argument('-w',              dest='weightsFn',      required = True,   help=' ')
parser.add_argument('-s',              dest='sType',          required = True,   help='train, test, dev')
parser.add_argument('-r',              dest='resultsFn',      default='',        help=' ')
parser.add_argument('-p',              dest='pickleFn',       default='',        help='store yvals and targets')

parser.add_argument('--gpuMem',        dest='gpuMem',         default=0.0,      type=float,                     help='0.0=no gpu, 1.0=all memory')
parser.add_argument('--hardSG',        dest='hardSG',         default=False,    type=distutils.util.strtobool,  help='force HardSG from soft')
parser.add_argument('--forceSenna',    dest='forceSenna',     default=False,    type=distutils.util.strtobool,  help='translate from Glove to Senna')
parser.add_argument('--forceGlove',    dest='forceGlove',     default=False,    type=distutils.util.strtobool,  help='translate from Senna to Glove')

parser.add_argument('--debug',         dest='debug',          default=False,    type=distutils.util.strtobool,  help='Debug')
parser.add_argument('--maxSamples',    dest='maxSamples',     default=None,     type=int,                       help='Maximum Samples from train, test, dev')
parser.add_argument('--noSG',          dest='noSG',           default=False,    type=distutils.util.strtobool,  help='no SG Feature input')

parser.add_argument('--testBatch',     dest='testBatch',      default=256,      type=int,                       help='batch size for test')



if len(sys.argv) == 1:
    # add default option string here
    aString =  ' '
    sys.argv = [''] +  aString.split(' ')
    print sys.argv

if sys.argv[1].startswith('@'):
    args, unknown = parser.parse_known_args()
    args, unknown = parser.parse_known_args( shlex.split(open(sys.argv[1][1:]).read()) )
    if unknown:
        print '\n' * 10
        print 'Warning, unknown args', unknown
        print '\n' * 10
else:
    args = parser.parse_args()
    
s = []
for arg in vars(args):
     s.append( '%-20s = %-20s      %-20s ' % (arg,  getattr(args, arg), '(' + str(type(getattr(args, arg))) + ')' ) )
s.sort()
#print '\n'.join(s)


if (args.gpuMem < 0.01):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

def get_session(gpu_fraction=0.6):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

if args.gpuMem >= 0.01:
    KTF.set_session(get_session(gpu_fraction=args.gpuMem))

import keras
from keras import backend as K
from keras.layers import Input, Embedding, LSTM, Dense, Reshape, merge, Concatenate
from keras.layers import Activation, Lambda, Dropout, Layer, Masking, TimeDistributed, Bidirectional
from keras.models import Model, model_from_json
from SGGenerator import *
from pprint import pprint as p
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model

                       
ga = AMRDataGenerator.getGeneralArch(args.vectorFn) 
if ga['target']=='L0':
    agt  = SGGenerator(args.vectorFn,  args.sType,     args.testBatch       , maxItems=args.maxSamples)
elif ga['target']=='args':
    agt  = ArgsGenerator(args.vectorFn,  args.sType,     args.testBatch       , maxItems=args.maxSamples)
elif ga['target']=='nargs':
    agt  = NargsGenerator(args.vectorFn,  args.sType,     args.testBatch       , maxItems=args.maxSamples)
elif ga['target']=='attr':
    agt  = AttrGenerator(args.vectorFn,  args.sType,     args.testBatch       , maxItems=args.maxSamples)
elif ga['target']=='ncat':
    agt  = CatGenerator(args.vectorFn,  args.sType,     args.testBatch       , maxItems=args.maxSamples)
else:
    print 'Type of network is not determined by the vector genArch:'
    p(ga)
    print ga
    exit(1)

# load json and create model
json_file = open(args.modelFn, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(args.weightsFn)
model.summary()
distSG_embedding_matrix = agt.readAMRDBFeatureInfo()
layers = model.layers
loadCount=0
for i in range(len(layers)):
    name = layers[i].name
    if 'distSGTable' == name:   
        print 'loading weights from vectors into model for ', name
        w = model.get_layer(name).get_weights()
        print w[0].shape
        print distSG_embedding_matrix.shape
        w[0][:distSG_embedding_matrix.shape[0]] = distSG_embedding_matrix
        model.get_layer(name).set_weights(w)
        loadCount+=1
    elif 'logDistSGTable' == name:   
        log_em = np.log(distSG_embedding_matrix + 1e-20)  # don't allow zero, log is inf.
        log_em[0] *= 0.0
        print 'loading weights from vectors into model for ', name
        w = model.get_layer(name).get_weights()
        print w[0].shape
        print log_em.shape
        w[0][:log_em.shape[0]] = log_em
        model.get_layer(name).set_weights(w)
        loadCount+=1
if loadCount != 1:
    print 'WARNING, load count is', loadCount
        



numberOfBatches =  (agt.numberOfItems())/args.testBatch
if numberOfBatches * args.testBatch < agt.numberOfItems():
    numberOfBatches += 1
y_vals  = model.predict_generator(agt.generate(), numberOfBatches)[0:agt.numberOfItems()]
agt.setCurrentIX(0)
targets = agt.getTargets(agt.numberOfItems() )
print 'yvals, targets: ', len(y_vals), len(targets)
if args.pickleFn:
    pickle.dump ( (y_vals, targets), open(args.pickleFn, 'wb') )
if args.resultsFn:
    df, sm, rc, sc,  precision, recall, f1, cString = agt.writeAMRResultsDatabase(args.resultsFn, y_vals, targets)
else:
    df, sm, rc, sc,  precision, recall, f1, cString = agt.getConfusionStats(y_vals, targets)
    
print cString
print df
print 'test sm, rc, sc, precision, recall, f1:', sm, rc, sc, precision, recall, f1
print 'Done'


