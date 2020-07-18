# -*- coding: utf-8 -*-
# Version: June 2 14:04:18 CEST 2017

import numpy as np
import sys
import csv
import os
import time
import pickle

import theanets
import theano

from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers


class AdjustVariable(object):
    '''
    Adjust variables for training
    '''
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)
sys.setrecursionlimit(10000) # increase recursion limit for saving the model


class AutoencoderStructure:
    '''
    Define Autoencoder structure. Attributes:
        n_input_vars: default is 60 for 20 weights per joint or DOF
        n_hid_layers: default is half which is 30
        act_function: default is tanh which is hyperbolic tangent
    '''
    def __init__(self, n_input_vars=60, n_lat_vars=3, act_function='tanh'):
        self.n_in_vars = n_input_vars # 20 weights per joint (DOF)
        self.n_lat_vars = n_lat_vars
        self.act_function = act_function

        l = [] # result
        if self.n_lat_vars >= 20:
            n_hid_layers = 3
            lspace = np.linspace(self.n_lat_vars, self.n_in_vars, n_hid_layers) # equidistant
            l.append(self.n_in_vars)
            l.append((int(lspace[1]), self.act_function))
            for d in lspace:
                l.append((int(d), self.act_function))
        else:
            n_hid_layers = 6
            lspace = np.linspace(self.n_lat_vars, self.n_in_vars, n_hid_layers-1) # -1 because linspace of only half
            l.append(self.n_in_vars)
            l.append((int(lspace[2]), self.act_function))
            l.append((int(lspace[1]), self.act_function))
            for d in lspace:
                l.append((int(d), self.act_function))
        self.ae_structure = tuple(l)

    # return the structure of the autoencoder
    # e.g.: (60 (45, 'tanh'), (30, 'tanh'), (45, 'tanh'), (60, 'tanh'))
    def get_ae_structure(self):
        return self.ae_structure

    def print_ae_structure(self):
        s = ""
        s += "("
        for i in range(len(self.ae_structure)):
            if i == 0:
                s += str(self.ae_structure[i]) + ", " # only int not tuple
            elif i < len(self.ae_structure) - 1:
                s += str(self.ae_structure[i][0]) + ", "
            else:
                s += str(self.ae_structure[i][0]) + ")"
        print(s)

    def count_hidden_layer_nodes(self):
        n_nodes = 0
        for i in range(1, len(self.ae_structure)-1):
            n_nodes += self.ae_structure[i][0]
        return n_nodes

    def count_links(self):
        n_links = 0
        for i in range(len(self.ae_structure)-1):
            if i == 0:
                n_links += self.ae_structure[i] * self.ae_structure[i+1][0]  # only int not tuple
            else:
                n_links += self.ae_structure[i][0] * self.ae_structure[i+1][0]
        return n_links


def prepare_sets(path, header):
    '''
    Read the feature matrix from csv file
    '''
    with open(path) as csvfile:
        if header:
            next(csvfile) # ignore header
        data = [row.strip().split(',') for row in csvfile]
    return data


def prepare_response(path, header):
    '''
    Reads the response vector from csv file
    '''
    with open(path) as csvfile:
        if header:
            next(csvfile) # ignore the header
        data = []
        for row in csvfile:
            data.append(int(row.strip().split(',')[0]))
    return data


def load_weights():
    path_train_x = '../data/train_weights_example.csv'
    path_valid_x = '../data/valid_weights_example.csv'
    path_test_x = '../data/test_weights_example.csv'

    train_set_x = np.asarray(prepare_sets(path_train_x, False), dtype=np.float32)
    valid_set_x = np.asarray(prepare_sets(path_valid_x, False), dtype=np.float32)
    test_set_x = np.asarray(prepare_sets(path_test_x, False), dtype= np.float32)

    return train_set_x, test_set_x, valid_set_x


def train_net_theano():
    train, valid, test = load_weights()

    print('Training AE')
    start = time.time()

    ae_struct = AutoencoderStructure()
    ae_struct.print_ae_structure()

    net = theanets.Autoencoder(layers = ae_struct.get_ae_structure())

    net.train(train,
               valid,
               algo='adadelta',#rmsprop
               # patience=.1,
               min_improvement=.01,
               #input_noise=.1,
               train_batches=1000,
               momentum=.9,
               weight_l2=.0001)

    end = time.time()
    print(np.linalg.norm(net.decode(net.encode(test)) - test))
    print("Elapsed time: ")
    print(round(end - start, 2))

    return net


def train_net_keras():
    train, valid, test = load_weights()

    print('Training AE')
    start = time.time()

    input_shape = Input(shape=(60,))
    encoded = Dense(units=45, activation='relu')(input_shape)
    encoded = Dense(units=31, activation='relu')(encoded)
    encoded = Dense(units=17, activation='relu')(encoded)
    encoded = Dense(units=3, activation='relu')(encoded)
    decoded = Dense(units=17, activation='relu')(encoded)
    decoded = Dense(units=31, activation='relu')(decoded)
    decoded = Dense(units=45, activation='relu')(decoded)
    decoded = Dense(units=60, activation='sigmoid')(decoded)

    net = Model(input_shape, decoded)
    encoder = Model(input_shape, encoded)

    net.compile(optimizer='adadelta',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

    net.fit(train,
             train,
             epochs=50,
             batch_size=1000,
             shuffle=True,
             validation_data=(valid, valid))

    end = time.time()
    print(np.linalg.norm(net.predict(test) - test))
    print("Elapsed time: ")
    print(round(end - start, 2))

    return net


def save_matrices(weights):
    for i in range(len(weights)):
        with open('../data/matrix-' + str(i+1) + '.csv', 'w') as wri:
            write = csv.writer(wri, delimiter = ';', quotechar ='"', quoting=csv.QUOTE_MINIMAL)
            for j in weights[i]:
                write.writerow(j)

THEANO = False
if THEANO:
    net = train_net_theano()
    ae_struct = AutoencoderStructure()
    save_matrices([net.find(i, 'w').get_value() for i in range(1, len(ae_struct.ae_structure))])
    # biases = [net.find(1, 'b').get_value() for i in range(1, len(ae_struct.ae_structure))]
else:
    net = train_net_keras()
    save_matrices([layer.get_weights() for layer in net.layers])
