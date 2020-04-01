#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
file:          NeuroNet.py
description:   definition of class <NeuroNet> representing generic artificial 
               neural networks with basic functionality
author:        Stefan Wittwer
e-mail:        info@wittwer-datatools.ch
"""


# %% import modules
from numpy import array, dot, random, transpose, zeros
from scipy import special


# %% definition of class <NeuroNet>
class NeuroNet:
    """
    The class <NeuroNet> represents a light-weight and customizable neural 
    network. Both the number of layers and the number of nodes of each layer 
    are customizable at constructing time.
    
    The activation of the neurons is implemented with the logistic function
                    1
        f(x) = -------------
                1 + exp(-x)
    
    Parameters:
    -----------
    E          list of error arrays
    N          array containing number of nodes of two or more layers
    S          signal array
    W          list of weight matrices
    r          learning rate in ]0; 1[
    """
    E = list()
    N = array([3, 3])
    S = list()
    W = list()
    r = 0.5

    # class methods
    def __init__(self, N: list, r = 0.5):
        """
        Standard constructor of <NeuroNet> objects.
        """
        # set number of neurons (nodes) in each layer
        if 1 < len(N):
            self.N = array(N, 'uint')
            if all(1 < self.N):
                for n in range(len(self.N)):
                    self.S.append(zeros((self.N[n], 1), 'float64'))
                    self.E.append(zeros((self.N[n], 1), 'float64'))
            else:
                self.N = list()
                raise(ValueError('Numbers of neurons must be integers > 1.'))
        else:
            raise(ValueError('Number of neuron layers must be > 1.'))
        # set learning rate
        if 0.0 < r and r < 1.0:
            self.r = r
        else:
            raise(ValueError('Learning rate must be in ]0.0; 1.0[.'))
        # initialize weight matrices
        for n in range(len(N) - 1):
            self.W.append(random.rand(self.N[n + 1], self.N[n]) - 0.5)

    def QueryNet(self, signal):
        # convert signal input to array
        self.S[0] = array(signal, ndmin = 2).T
        #propagate signal array through all layers using logistic activation
        for s, w in enumerate(self.W):
            self.S[s + 1] = special.expit(dot(w, self.S[s]))
        # return output layer signals
        return self.S[-1]

    def TestNet(self, signal, target):
        error = self.TrainNet(signal, target);
        return error

    def TrainNet(self, signal, target):
        # convert target argument to array
        target = array(target, ndmin = 2).T
        # query neural network with signal input
        self.QueryNet(signal)
        # initialize list of error arrays
        n = len(self.N) - 1
        self.E[-1] = target - self.S[-1]
        # propagate error back
        for i in range(len(self.E) - 1):
            self.E[n - i - 1] = dot(self.W[n - i - 1].T, self.E[n - i])
        #train weights
        for i, w in enumerate(self.W):
            a = self.E[i + 1] * self.S[i + 1] * (1.0 - self.S[i + 1])
            b = self.S[i].T
            w += self.r * dot(a, b)      
        #return errors
        return self.E


# %% end of module <NeuroNet>
