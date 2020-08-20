import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

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
from tensorflow.compat.v1.keras import backend as K

import datetime
import math
import hashlib
import time
import os
from datagenerator import DataGenerator
# from datagenerator_b import DataGenerator
import random

# import matplotlib.pyplot as plt # add
# add # Plot and save fig without displaying it on X server 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

seed_value= 10

#Disable randomisation
os.environ['PYTHONHASHSEED']=str(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

session_conf.gpu_options.allow_growth = True # add

sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)

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

    return [x], [y]


def getTokenizer(df) :
    ### Dictionary for Normal ###
    tokenizer = Tokenizer(filters='', split='<sep>', oov_token='<OTHERS>' ,lower=True)
    tokenizer.fit_on_texts(df['Input'].values)

    tokenizer.fit_on_texts(['<SOS>'])
    tokenizer.fit_on_texts(['<EOS>'])

    return tokenizer

def createGeneratorData(df, tokenizer, max_len) :
    #Prepare training for normal model
    x = []
    y = []

    for seq in df['Input']:
        x_windows, y_windows = prepare_sentence(seq, max_len, tokenizer)
        x += x_windows
        y += y_windows
    x = np.array(x)
    y = np.array(y)  # The word <PAD> does not constitute a class

    x.shape = [len(x), max_len + 1, 1]
    y.shape = [len(y), max_len + 1, 1]

    return x, y

def trainModelP(size, max_len, vocab_size, config):
    # Parameters
    params = {'dim': (max_len + 1, 1),
          'batch_size': config['TRAININGPARAMS']['BATCH_SIZE'],
          'progress' : config['metadata']['uniqueID'],
          'n_classes': vocab_size,
          'n_channels': 1,
          'shuffle': True}

    input_emb_dim = config['MODELPARAMS']['INPUT_EMBED_DIM']
    lstm_emb_dim = config['MODELPARAMS']['LSTM_DIM']
    # Datasets generation
    cut = math.ceil(size*config['TRAININGPARAMS']['PERCENTAGETRAIN'])
    partition = dict()
    partition['train'] = []
    partition['validation'] = []

    for i in range(cut):
        partition['train'].append(i)

    for i in range(cut, size) :
        partition['validation'].append(i)

    # Generators
    training_generator_uri_normal = DataGenerator(False, None, partition['train'], **params)
    validation_generator_uri_normal = DataGenerator(False, None, partition['validation'], **params)

    #Input Model for Combined Sequences
    visible1 = Input(shape=(max_len + 1, 1), name = 'Input')

    embedded1 = Embedding(vocab_size[0] + 1, input_emb_dim, input_length = (max_len + 1, 1), name = 'Embedding')(visible1)

    embedded1 = Reshape((max_len + 1, input_emb_dim))(embedded1)

    merge = embedded1
    # add Dropout
    merge = LSTM(lstm_emb_dim, return_sequences=True, dropout=0.2, name = 'LSTM1')(merge)
    merge = LSTM(lstm_emb_dim, return_sequences=True, dropout=0.2, name = 'LSTM2')(merge)

    #Output Model for Combined Sequences
    # output1 = TimeDistributed(Dense(vocab_size[0] + 1, activation='softmax'), name='Output')(merge)
    output1 = Dense(vocab_size[0] + 1, activation='softmax', name='Output')(merge) # add

    modelP = Model(inputs = [visible1], outputs = [output1])
    adam = optimizers.adam(lr=config['MODELPARAMS']['LEARNING_RATE_P'])
    modelP.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])

    print(modelP.summary())

    historyP = modelP.fit_generator(generator=training_generator_uri_normal,
                        validation_data=validation_generator_uri_normal,
                        use_multiprocessing = config['TRAININGPARAMS']['MULTIPROCESSING'],
                        workers = config['TRAININGPARAMS']['WORKERS'],
                        epochs = config['TRAININGPARAMS']['EPOCHS'],
                        callbacks=[EarlyStopping(monitor='val_loss',patience=9, mode='auto', min_delta=0.0001)])

    return modelP, historyP # add


def trainModelQ(size, max_len, vocab_size, config, fname, index=0, online=True):
    # Parameters
    params = {'dim': (max_len + 1, 1),
          'batch_size': config['TRAININGPARAMS']['BATCH_SIZE'], #math.floor(size*config['TRAININGPARAMS']['PERCENTAGETRAIN']/3),
          'progress' : config['metadata']['uniqueID'],
          'n_classes': vocab_size,
          'n_channels': 1,
          'shuffle': True}

    input_emb_dim = config['MODELPARAMS']['INPUT_EMBED_DIM']
    lstm_emb_dim = config['MODELPARAMS']['LSTM_DIM_Q']
    # Datasets generation
    cut = math.ceil(size*config['TRAININGPARAMS']['PERCENTAGETRAIN'])
    partition = dict()
    partition['train'] = []
    partition['validation'] = []

    for i in range(cut):
        partition['train'].append(i)

    for i in range(cut, size) :
        partition['validation'].append(i)

    # #Generators
    # training_generator_uri_attack = DataGenerator(True, 0, partition['train'], **params)
    # validation_generator_uri_attack = DataGenerator(True, 0, partition['validation'], **params)
    training_generator_uri_attack = DataGenerator(True, index, partition['train'], **params)
    validation_generator_uri_attack = DataGenerator(True, index, partition['validation'], **params)

    #Change name if not transfering all layer. Subsequent loading of weights wont load if name is different
    ModelName = []
    if config['TRAININGPARAMS']['ALLLAYERTRANSFER'] :
        ModelName.append('LSTM1')
        ModelName.append('LSTM2')
        ModelName.append('Output')
    else :
        ModelName.append('LSTM1Q')
        ModelName.append('LSTM2Q')
        ModelName.append('OutputQ')


    #Input Model for Combined Sequences
    visible1 = Input(shape=(max_len + 1, 1), name = 'Input')
    embedded1 = Embedding(vocab_size[0] + 1, input_emb_dim, input_length = (max_len + 1, 1), name = 'Embedding')(visible1)
    embedded1 = Reshape((max_len + 1, input_emb_dim))(embedded1)

    merge = embedded1
    # add Dropout
    merge = LSTM(lstm_emb_dim, return_sequences=True, dropout=0.0, name = ModelName[0])(merge)

    if config['TRAININGPARAMS']['2LAYERLSTMQ'] :
        merge = LSTM(lstm_emb_dim, return_sequences=True, dropout=0.0, name = ModelName[1])(merge)

    #Output Model for Combined Sequences
    # output1 = TimeDistributed(Dense(vocab_size[0] + 1, activation='softmax'), name = ModelName[2])(merge)
    output1 = Dense(vocab_size[0] + 1, activation='softmax', name = ModelName[2])(merge)

    modelQ = Model(inputs = [visible1], outputs = [output1])

    if fname is not None :
        # modelQ.load_weights(fname, by_name=True, skip_mismatch=True)
        modelQ.load_weights((config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + fname), by_name=True, skip_mismatch=True)

    adam = optimizers.adam(lr=config['MODELPARAMS']['LEARNING_RATE_Q'])
    modelQ.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])

    print(modelQ.summary())

    # if config['ONLINETRAINING'] :
    if online:
        epoch = config['TRAININGPARAMS']['INTERVALEPOCHS']
        callback = []
    else :
        epoch = config['TRAININGPARAMS']['EPOCHS']
        callback = [EarlyStopping(monitor='val_loss',patience=9, mode='auto', min_delta=0.0001)]

    historyQ = modelQ.fit_generator(generator=training_generator_uri_attack,
                        validation_data=validation_generator_uri_attack,
                        use_multiprocessing = config['TRAININGPARAMS']['MULTIPROCESSING'],
                        workers = config['TRAININGPARAMS']['WORKERS'],
                        epochs = epoch,
                        callbacks = callback)

    lastLoss = historyQ.history['loss'][-1] #[2] # add
    lastacc = historyQ.history['acc'][-1] # add

    return modelQ, lastLoss, lastacc, historyQ # add

def updateModel(model, index, size, max_len, vocab_size, config, fname) :
    # Parameters
    params = {'dim': (max_len + 1, 1),
          'batch_size': config['TRAININGPARAMS']['BATCH_SIZE'], #math.floor(size*config['TRAININGPARAMS']['PERCENTAGETRAIN']/3),
#          'batch_size': min(math.floor(size*config['TRAININGPARAMS']['PERCENTAGETRAIN']), 8),
          'progress' : config['metadata']['uniqueID'],
          'n_classes': vocab_size,
          'n_channels': 1,
          'shuffle': True}

    input_emb_dim = config['MODELPARAMS']['INPUT_EMBED_DIM']
    lstm_emb_dim = config['MODELPARAMS']['LSTM_DIM_Q']
    # Datasets generation
    cut = math.floor(size*config['TRAININGPARAMS']['PERCENTAGETRAIN'])
    partition = dict()
    partition['train'] = []
    partition['validation'] = []

    for i in range(cut):
        partition['train'].append(i)

    for i in range(cut, size) :
        partition['validation'].append(i)

    # Generators
    training_generator_uri_attack = DataGenerator(True, index, partition['train'], **params)
    validation_generator_uri_attack = DataGenerator(True, index, partition['validation'], **params)

    historyQ = model.fit_generator(generator=training_generator_uri_attack,
                        validation_data=validation_generator_uri_attack,
                        use_multiprocessing = config['TRAININGPARAMS']['MULTIPROCESSING'],
                        workers = config['TRAININGPARAMS']['WORKERS'],
                        epochs = config['TRAININGPARAMS']['INTERVALEPOCHS'])

    lastLoss = historyQ.history['loss'][-1] #[2] # add
    lastacc = historyQ.history['acc'][-1] # add

    return model, lastLoss, lastacc

def main():
    print("*****     Starting Training     ******")
    config = loadConfig()
    max_len = config['SEQUENCELENGTH']

    # print("*****     Preparing Model P     ******")

    df_normal = pd.read_csv(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'N1.csv')
    tokenizer_normal = getTokenizer(df_normal)
    df_normal_embedded = df_normal.copy()
    df_normal_embedded['Input'] = tokenizer_normal.texts_to_sequences(df_normal['Input'].values)
    x_normal, y_normal = createGeneratorData(df_normal_embedded, tokenizer_normal, max_len)

    np.save(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'normalURITraining.npy', x_normal)
    np.save(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'normalURILabel.npy', y_normal)


    print("*****     Training Model P     ******")
    fname = "weights"
    start = time.time()
    # add historyP
    modelP, historyP = trainModelP(len(df_normal), max_len, [len(tokenizer_normal.word_index)], config)
    # modelP.save_weights(fname)
    modelP.save_weights(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + fname)
    print("Time to train Mode P is " + str(time.time() - start))

    # add =========== Plot training acc and loss ===========
    acc = historyP.history['acc']
    val_acc = historyP.history['val_acc']
    loss = historyP.history['loss']
    val_loss = historyP.history['val_loss']
    epochs = range(1, len(acc) + 1)
    f = plt.figure(figsize=(15,5))

    ax1 = f.add_subplot(121)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.plot(epochs, acc, 'bo', label='Training acc')
    ax1.plot(epochs, val_acc, 'b', label='Validation acc')
    ax1.legend()
    ax1.title.set_text('Training and validation accuracy')

    ax2 = f.add_subplot(122)
    ax2.title.set_text('Generated Synthetic Graph')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.plot(epochs, loss, 'bo', label='Training loss')
    ax2.plot(epochs, val_loss, 'b', label='Validation loss')
    ax2.legend()
    ax2.title.set_text('Training and validation loss')
    f.savefig('./'+ config['metadata']['uniqueID'] +'/'+ config['metadata']['result'] + '_Ptraining.png')




    print("*****     Preparing Model Q     ******")

    # Online Q interval data
    count = 0
    lengths = []
    while(True) :
        if not os.path.isfile(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'A1_' + str(count) + '.csv') :
            break
        df_attack = pd.read_csv(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'A1_' + str(count) + '.csv')
        tokenizer_attack = tokenizer_normal
        df_attack_embedded = df_attack.copy()
        df_attack_embedded['Input'] = tokenizer_attack.texts_to_sequences(df_attack['Input'].values)
        x_attack, y_attack = createGeneratorData(df_attack_embedded, tokenizer_attack, max_len)
        np.save(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'attackURITraining_' + str(count) + '.npy', x_attack)
        np.save(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'attackURILabel_' + str(count) + '.npy', y_attack)
        lengths.append(len(df_attack))
        count = count + 1

    # Offline Q data
    df_attackfull = pd.read_csv(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'A1_full.csv')
    tokenizer_attack = tokenizer_normal
    df_attackfull_embedded = df_attackfull.copy()
    df_attackfull_embedded['Input'] = tokenizer_attack.texts_to_sequences(df_attackfull['Input'].values)
    x_attackfull, y_attackfull = createGeneratorData(df_attackfull_embedded, tokenizer_attack, max_len)
    np.save(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'attackURITraining_full.npy', x_attackfull)
    np.save(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'attackURILabel_full.npy', y_attackfull)



    print("*****     Training Model Q  (Online with Embedding Transfer)   ******")
    timeArrayWithEmbed = []
    lossArrayWithEmbed = []
    accArrayWithEmbed = []

    start = time.time()
    # add historyQT 
    modelQT, loss, acc, _ = trainModelQ(lengths[0], max_len, [len(tokenizer_attack.word_index)], config, fname)
    print("Time to train Mode Q is " + str(time.time() - start))
    timeArrayWithEmbed.append(time.time() - start)
    lossArrayWithEmbed.append(loss)
    accArrayWithEmbed.append(acc)

    print("*****     Updating Model Q  (With Embedding Transfer)   ******")
    for i in range(1, count, 1) :
        print("*****     Updating Model Q  with time interval " + str(i) + "   ******")

        start = time.time()
        modelQT, loss, acc = updateModel(modelQT, i, lengths[i], max_len, [len(tokenizer_attack.word_index)], config, fname)
        timeA = time.time() - start
        timeArrayWithEmbed.append(timeA)
        lossArrayWithEmbed.append(loss)
        accArrayWithEmbed.append(acc)




    print("*****     Training Model Q  (Offline no transfer)   ******")

    start = time.time()
    # add historyQ 
    modelQ, loss, _, historyQ = trainModelQ(len(df_attackfull), max_len, [len(tokenizer_attack.word_index)], config, None, index=None, online=False)
    print("Time to train Mode Q is " + str(time.time() - start))

    # add =========== Plot training acc and loss ===========
    acc = historyQ.history['acc']
    val_acc = historyQ.history['val_acc']
    loss = historyQ.history['loss']
    val_loss = historyQ.history['val_loss']
    epochs = range(1, len(acc) + 1)
    f = plt.figure(figsize=(15,5))

    ax1 = f.add_subplot(121)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.plot(epochs, acc, 'bo', label='Training acc')
    ax1.plot(epochs, val_acc, 'b', label='Validation acc')
    ax1.legend()
    ax1.title.set_text('Training and validation accuracy')

    ax2 = f.add_subplot(122)
    ax2.title.set_text('Generated Synthetic Graph')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.plot(epochs, loss, 'bo', label='Training loss')
    ax2.plot(epochs, val_loss, 'b', label='Validation loss')
    ax2.legend()
    ax2.title.set_text('Training and validation loss')
    # f.savefig('./'+ config['metadata']['uniqueID']+'/train_Q_1lstmlayer.png')
    f.savefig('./'+ config['metadata']['uniqueID'] +'/'+ config['metadata']['result'] + '_offlineQtraining.png')


    # # #Save loss/time taken data in output.csv
    # df = pd.DataFrame(data={"col1":timeArrayWithEmbed, "col2":accArrayWithEmbed , "col3":lossArrayWithEmbed})
    # df.to_csv(config['metadata']['uniqueID'] + '/' + config['metadata']['result'] + 'onlineQtraining_output.csv', sep=',',index=False)

    modelP.save(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'modelP')
    modelQT.save(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'modelQWithTransfer')
    modelQ.save(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'modelQWithoutTransfer')

    # saving normal
    with open(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'tokenizer_normal.pickle', 'wb') as handle:
        pickle.dump(tokenizer_normal, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # saving attack
    with open(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'tokenizer_attack.pickle', 'wb') as handle:
        pickle.dump(tokenizer_attack, handle, protocol=pickle.HIGHEST_PROTOCOL)

    K.clear_session()


    print("*****     Ending Training     ******")

if __name__ == "__main__":
    main()
