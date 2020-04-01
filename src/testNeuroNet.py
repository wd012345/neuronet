#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
file:          testNeuroNet.py
description:   simple test for objects of the <NeuroNet> class
author:        Stefan Wittwer
e-mail:        info@wittwer-datatools.ch
"""


# %% import modules
from NeuroNet import NeuroNet
from numpy import argmax, asfarray, linalg


# %% set test parameters
N = [12, 9, 5]         # layer structure
N_e = 24;              # number of epochs
Error = list()         # norm of output error after each epoch
r = 0.60               # learning rate (default: 0.5)
score = list()         # list with scores


# %% prepare training and test signals
signs = ['0', '1', '.', 'x', '+']
test_signs = ['1', 'x', '0']
signals = [
    [0, 1, 0, 
     1, 0, 1, 
     1, 0, 1, 
     0, 1, 0], 
    [0, 1, 0, 
     1, 1, 0, 
     0, 1, 0, 
     0, 1, 0, ], 
    [0, 0, 0, 
     0, 0, 0, 
     0, 0, 0, 
     0, 1, 0], 
    [0, 0, 0, 
     1, 0, 1, 
     0, 1, 0,
     1, 0, 1], 
    [0, 0, 0, 
     0, 1, 0, 
     1, 1, 1,
     0, 1, 0]]
targets = [
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0], 
    [0, 0, 1, 0, 0], 
    [0, 0, 0, 1, 0], 
    [0, 0, 0, 0, 1]]
tests = [
    [0, 1, 0, 
     1, 1, 0, 
     0, 1, 0, 
     0, 1, 0], 
    [0, 0, 0, 
     1, 0, 1, 
     0, 1, 0,
     1, 0, 1], 
    [0, 1, 0, 
     1, 0, 1, 
     1, 0, 1, 
     0, 1, 0]]


# %% construct and train <NeuroNet> object
nn = NeuroNet(N, r)
for n_e in range(N_e):
    for k, signal in enumerate(signals):
        # scale signals to range [0.01; 0.99]
        signal = asfarray(signal) * 0.98 + 0.01
        # scale targets to [0.01; 0.99]
        target = asfarray(targets[k]) #* 0.98 + 0.01
        #launch trainer
        error = nn.TrainNet(signal, target)
    Error.append(linalg.norm(error[-1]))
    print('processed epoch {0} with error {1}'.format(n_e, Error[n_e]))


# %% test neural network using test data
for k, signal in enumerate(tests):
    #scale and shift inputs
    signal = asfarray(signal) * 0.98 + 0.01
    #query network
    output = nn.QueryNet(signal)
    #print result
    print('test {0}: {1}'.format(k, signs[argmax(output)]))
    score.append(int(test_signs[k] == signs[argmax(output)]))
print(score)


# %% end of module <testNeuroNet>
