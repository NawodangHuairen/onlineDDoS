import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
import tensorflow as tf
import pickle
import sys
import yaml

from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Flatten, TimeDistributed, Bidirectional
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
from keras.utils import plot_model
from keras import optimizers
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.models import load_model
from sklearn import preprocessing
from copy import deepcopy

import datetime
import math
import hashlib
import time
import os
from datagenerator import DataGenerator

# add
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True 
# # sess = tf.Session(config=config) 
sess = tf.compat.v1.Session(config=config) 


def loadConfig():
    with open(sys.argv[1], "r") as ymlfile:
        cfg = yaml.load(ymlfile)
    return cfg

def prepare_sentence(seq, maxlen, tokenizer):
    # Pads seq and slides windows
    seq = seq[:maxlen]
    seqX = np.append(tokenizer.word_index['<sos>'], seq)
    seqY = np.append(seq, tokenizer.word_index['<eos>'])

    x= pad_sequences([seqX],
        maxlen=maxlen+1,
        padding='post')[0]  # Pads before each sequence

    y= pad_sequences([seqY],
        maxlen=maxlen+1,
        padding='post')[0]  # Pads before each sequence

    return x, y

# Compute probability of occurence of a sentence
def seqPrepX(sentence, pair, model, tokenizer, histogram, datasize, maxlen) :
    #Assigns indices to each request
    tok = tokenizer.texts_to_sequences([str(sentence)])[0]

    #Prepare sentence to get input and actual output of array
    #x_test and y_test are lists of length 21.
    x_test, y_test = prepare_sentence(tok, maxlen, tokenizer)

    x_test.shape = [maxlen + 1 ,1]

    return x_test


def seqPrepY(sentence, pair, model, tokenizer, histogram, datasize, maxlen) :
    #Assigns indices to each request
    tok = tokenizer.texts_to_sequences([str(sentence)])[0]

    #Prepare sentence to get input and actual output of array
    #x_test and y_test are lists of length 21.
    x_test, y_test = prepare_sentence(tok, maxlen, tokenizer)

    return y_test

def calcProbability(maxlen, p_pred_normal, y_test)  :
    #Chain rule to calculate probability of sentence
    p_sentence = 0
    for i in range(maxlen + 1) :
        if (y_test[i] != 0) :
            # p_sentence = p_sentence + math.log(p_pred_normal[i][y_test[i]])
            p_sentence = p_sentence + math.log(p_pred_normal[i]) # add

    return p_sentence

def calcHisto(pair, agent, country, histogram, agenthistogram, countryhistogram, datasize):
    #Get probability from Histogram
    p_hist = getHistogramProbability(histogram, pair, datasize)

    #Get probability from agent (Smoothed)
    agentMiniNum = agenthistogram.min()
    p_agent = getHistogramSmoothedProbability(agenthistogram, agent, datasize, agentMiniNum)

    #Get probability from country (Smoothed)
    countryMiniNum = countryhistogram.min()
    p_country = getHistogramSmoothedProbability(countryhistogram, country, datasize, countryMiniNum)

    p_hist_new = p_agent * p_country
    final_p_hist = (p_hist + p_hist_new)/2

    if final_p_hist == 0 :
        print(p_hist)
        print(p_agent)
        print(p_country)

    return math.log(final_p_hist)

# This might give the probability as 0
def getHistogramProbability(histogram, value, datasize):
    if value not in histogram :
        return 0
    else :
        return histogram[value]/datasize


# This ensures that there is at least some probability
def getHistogramSmoothedProbability(histogram, value, datasize, minimumNum):
    if value not in histogram :
        return (minimumNum/2)/datasize
    else :
        return histogram[value]/datasize

#Create dictionary such that we have the following information
#Dictionary  =
#{
#Create dictionary such that we have the following information
#Dictionary  =
#{
# '157.123.212.212': [{'sequence' : 'xx'
#                     'PoverQScore' : 12},]
#
# }
#}

#LOOK AT ATTACK AND SEE HOW CONFUSION MATRIX
#def cal_poverq(p,q):
#    if q == 0 :
#        print(q)
#        poverq = math.inf
#    else :
#        poverq = p/q
#    return poverq

#def cal_poverqHisto(p, qHisto):
#    if qHisto == 0 :
#        print(qHisto)
#        poverqHisto = math.inf
#    else :
#        poverqHisto = p/qHisto
#    return poverqHisto


# add #Function to convert 3d prediction array into 2d list using real sequences
def pred2list(Pred, df_Py):
    # convert Py to 2D array
    Py = df_Py.tolist() # add
    Py = np.vstack(Py) # (9847, 121) # add
    
    # Mask all zeros
    Pred_mask = np.zeros(Pred.shape, dtype=bool)
    Pred_mask[:,:,:] = Py[:,:,np.newaxis] == 0
    field3d = np.ma.array(Pred, mask=Pred_mask)
    
    # select elements from 3d array using 2d indices
    v_shp = field3d.shape #(9847, 121, 1008)
    y,x = np.ogrid[0:v_shp[0], 0:v_shp[1]]
    field3d = field3d[y, x, Py]#.shape
    field3d = field3d.filled(0) # fill mask=True with zeros
#     field3d.shape #(9847, 121)
    field3d = field3d.tolist()
    
    return field3d

#Attack Scores
def calculateSequenceScores(modelP, modelQWithT, modelQWithoutT, tokenizer_normal, tokenizer_attack, df_normal, df_attack, df_test, maxlen):
# def calculateSequenceScores(modelP, tokenizer_normal, df_normal, df_test, maxlen):    # add for P only
# def calculateSequenceScores(modelP, modelQWithT, tokenizer_normal, tokenizer_attack, df_normal, df_attack, df_test, maxlen):    # add for Q no transfer
    scoreDictionary = dict()
    count = 0
    df_temp = df_test[['Source IP','Dest IP', 'Input', 'Histo', 'Protocols', 'Info']].copy()

    norm_size = len(df_normal)
    atk_size = len(df_attack)

    CountryAgentHistogram_Normal = df_normal['Histo'].value_counts()
    CountryAgentHistogram_Attack = df_attack['Histo'].value_counts()

    AgentHistogram_Normal = df_normal['Protocols'].value_counts()
    AgentHistogram_Attack = df_attack['Protocols'].value_counts()

    CountryHistogram_Normal = df_normal['Info'].value_counts()
    CountryHistogram_Attack = df_attack['Info'].value_counts()


    print('c1')
    df_temp['Px'] = df_temp.apply(lambda x: seqPrepX(x["Input"], x["Histo"], modelP, tokenizer_normal, CountryAgentHistogram_Normal, norm_size, maxlen), axis=1)
    df_temp['Py'] = df_temp.apply(lambda x: seqPrepY(x["Input"], x["Histo"], modelP, tokenizer_normal, CountryAgentHistogram_Normal, norm_size, maxlen), axis=1)

    print('c2')
    df_temp['Qx'] = df_temp.apply(lambda x: seqPrepX(x["Input"], x["Histo"], modelP, tokenizer_attack, CountryAgentHistogram_Attack, atk_size, maxlen),axis=1)
    df_temp['Qy'] = df_temp.apply(lambda x: seqPrepY(x["Input"], x["Histo"], modelP, tokenizer_attack, CountryAgentHistogram_Attack, atk_size, maxlen),axis=1)

    print("Start inference")
    start = time.time()

    Pint = modelP.predict(np.array(df_temp['Px'].tolist()), batch_size=1024, verbose=1) # add 
    QWithTint = modelQWithT.predict(np.array(df_temp['Qx'].tolist()), batch_size=1024, verbose=1) # add
    QWithoutTint = modelQWithoutT.predict(np.array(df_temp['Qx'].tolist()), batch_size=1024, verbose=1) # add

    Pint_max = pred2list(Pint, df_temp['Py']) # add
    QWithTint_max = pred2list(QWithTint, df_temp['Qy']) # add
    QWithoutTint_max = pred2list(QWithoutTint, df_temp['Qy']) # add
    df_temp['Pint'] = Pint_max # add
    df_temp['QWithTint'] = QWithTint_max # add
    df_temp['QWithoutTint'] = QWithoutTint_max # add

    df_temp['P_LSTM'] = df_temp.apply(lambda x: calcProbability(maxlen, x['Pint'], x['Py']),axis=1)
    df_temp['hist1'] = df_temp.apply(lambda x: calcHisto(x["Histo"], x['Protocols'], x["Info"], CountryAgentHistogram_Normal, CountryHistogram_Normal, AgentHistogram_Normal, norm_size),axis=1)

    df_temp['QWithT_LSTM'] = df_temp.apply(lambda x: calcProbability(maxlen, x['QWithTint'], x['Qy']),axis=1)
    df_temp['hist2'] = df_temp.apply(lambda x: calcHisto(x["Histo"], x["Protocols"], x["Info"], CountryAgentHistogram_Attack, CountryHistogram_Attack, AgentHistogram_Attack, atk_size),axis=1)
    df_temp['QWithoutT_LSTM'] = df_temp.apply(lambda x: calcProbability(maxlen, x['QWithoutTint'], x['Qy']),axis=1)

    #Sorting to avoid mixing of IP and Scores
    df_temp = df_temp.sort_values(by = ['P_LSTM'])#, ignore_index = True)
    df_temp = df_temp.reset_index(drop=True) # add

    # add #No Scaling 
    df_min_max = deepcopy(df_temp[['P_LSTM','hist1', 'QWithT_LSTM', 'QWithoutT_LSTM' ,'hist2']]) #, 'QTilde']])
    df_min_max['Source IP'] = df_temp['Source IP']
    df_min_max['Dest IP'] = df_temp['Dest IP']

    # Computation of scores 
    df_min_max = df_min_max.rename(columns={0: 'P_LSTM', 1: 'hist1', 2: 'QWithT_LSTM', 3: 'QWithoutT_LSTM', 4: 'hist2'}) #, 5: 'QTilde'})
    df_min_max['P'] = df_min_max['P_LSTM'] + df_min_max['hist1']
    df_min_max['QWithT'] = df_min_max['QWithT_LSTM'] + df_min_max['hist2']
    df_min_max['QWithoutT'] = df_min_max['QWithoutT_LSTM'] + df_min_max['hist2']
    df_min_max['PoverQWithTransfer'] = df_min_max['P'] - df_min_max['QWithT']
    df_min_max['PoverQWithoutTransfer'] = df_min_max['P'] - df_min_max['QWithoutT']

    #Calculate time taken for inference
    print("End inference in {:.2f} seconds".format(time.time() - start))

    sr = df_min_max[['Source IP', 'Dest IP', 'P', 'PoverQWithTransfer','PoverQWithoutTransfer', 'QWithT', 'QWithoutT']] #, 'PoverQTilde']]
    # print('c4')
    sr.reset_index(inplace=True)
    # print('/////////////////')
    return sr


def main():
    print("*****     Starting Inferencing     ******")
    config = loadConfig()

    # #load models
    modelP = load_model(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'modelP')
    modelQWithTransfer = load_model(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'modelQWithTransfer')
    modelQWithoutTransfer = load_model(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'modelQWithoutTransfer')

    # load dataset
    df_normal = pd.read_csv(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'N1.csv')

    count = 0
    df_atk_arr = []
    while(True) :
        if not os.path.isfile(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'A1_' + str(count) + '.csv') :
            break

        df_attack = pd.read_csv(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'A1_' + str(count) + '.csv')
        df_atk_arr.append(df_attack)
        count = count + 1

    df_attack = pd.concat(df_atk_arr)

    # load testset
    df_test = pd.read_csv(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'TEST.csv')

    # loading tokenizer of normal
    with open(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'tokenizer_normal.pickle', 'rb') as handle:
        tokenizer_normal = pickle.load(handle)

    # loading tokenizer of attack
    with open(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'tokenizer_attack.pickle', 'rb') as handle:
        tokenizer_attack = pickle.load(handle)

    sr = calculateSequenceScores(modelP, modelQWithTransfer, modelQWithoutTransfer, tokenizer_normal, tokenizer_attack, df_normal, df_attack, df_test, config['SEQUENCELENGTH'])
    # sr = calculateSequenceScores(modelP, tokenizer_normal, df_normal, df_test, config['SEQUENCELENGTH']) # add for P only
    # sr = calculateSequenceScores(modelP, modelQWithTransfer, tokenizer_normal, tokenizer_attack, df_normal, df_attack, df_test, config['SEQUENCELENGTH']) # add for Q no transfer

    sr_P = sr[['Source IP', 'Dest IP', 'P']].copy()
    sr_PoverQWithTransfer = sr[['Source IP', 'Dest IP', 'PoverQWithTransfer']].copy()
    sr_PoverQWithoutTransfer = sr[['Source IP', 'Dest IP', 'PoverQWithoutTransfer']].copy()
    sr_onlineQ = sr[['Source IP', 'Dest IP', 'QWithT']].copy() # add for online Q 
    sr_offlineQ = sr[['Source IP', 'Dest IP', 'QWithoutT']].copy() # add for offline Q

    sr_P = sr_P.sort_values(by = ['P'])
    sr_PoverQWithTransfer = sr_PoverQWithTransfer.sort_values(by = ['PoverQWithTransfer'])
    sr_PoverQWithoutTransfer = sr_PoverQWithoutTransfer.sort_values(by = ['PoverQWithoutTransfer'])
    sr_onlineQ = sr_onlineQ.sort_values(by = ['QWithT']) # add for online Q 
    sr_offlineQ = sr_offlineQ.sort_values(by = ['QWithoutT']) # add for offline Q

    # #Store the data as binary data stream
    # sr_P.to_pickle(config['metadata']['uniqueID'] + '/' + config['metadata']['result'] + '/' + 'PScore')
    sr_P.to_pickle(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'PScore')

    # sr_PoverQWithTransfer.to_pickle(config['metadata']['uniqueID'] + '/' + config['metadata']['result'] + '/' + 'POverQWithTransferScore')
    sr_PoverQWithTransfer.to_pickle(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'POverQWithTransferScore')
    
    # sr_PoverQWithoutTransfer.to_pickle(config['metadata']['uniqueID'] + '/' + config['metadata']['result'] + '/' + 'POverQWithoutTransferScore')
    sr_PoverQWithoutTransfer.to_pickle(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'POverQWithoutTransferScore')

    sr_onlineQ.to_pickle(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'onlineQ')
    sr_offlineQ.to_pickle(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'offlineQ')

    print("*****     Ending Inferencing     ******")

    sess.close() # add

if __name__ == "__main__":
    main()
