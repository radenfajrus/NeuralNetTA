from __future__ import print_function

import os
import sys
import timeit
import numpy as np 
import Theano
import theano.tensor as T

class HiddenLayer(object):
	def _init__(self, rng, input, n_in,n_out, W = None, b=None,activation = T.tanh):

		self.input = input

		if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value =W_values, name= 'W', borrow=True)
        if b is None:
        	b_values = np.zeros((n_out,) ,dtype = theano.config.floatX)
        	b 		 = theano.shared(value=b_values,name='b', borow= True)
        self.W = W
        self.b = b
        
        lin_output = T.dot(input, self.w) + (self.b)
        self.output = (
        	lin_output if activation is None
        	else activation(lin_output)
        	)
        self.params= [self.W,self.b]
class HiddenLayer2(object):
	def _init__(self, rng, input, n_in,n_out, W=None,b=None,activation=T.tanh):
		self.input = input

		if W is None:
			W_values = np.asarray(
				rng.uniform(
					low=-numpy.sqrt(6./(n_in + n_out)),
					high=numpy.sqrt(6./(n_in + n_out)),
					size=(n_in, n_out)
				),
				dtype=theano.config.floatX
				)
			)
			if activation == theano.tensor.nnet.sigmoid:
					W_values *=4
			W = theano.shared(value=W_values, name='W', borrow=True)
		if b is None :
			b_values = np.zeros((n_out,),dtype = theano.config.floatX)
			b 		 = theano.shared(value=b_values, name='b', borrow= True)
		self.W = W
		self.b = b

		lin_output=T.dot(input, self.W) +(self.b)
		self.output=(
			lin_output if activation is None
			else activation(lin_output)
			)
		self.params = [self.W,self.b]
class MLP(object):
	def __init__(self, rng, input, n_in,n_out,n_hidden1,n_hidden2):

		self.hiddenLayer = HiddenLayer(
			rng = rng,
			input = input,
			n_in = n_in,
			n_out= n_hidden,
			activation=T.tanh
		)
		self.hiddenLayer2=HiddenLayer2(
			rng = rng,
			input=self.hiddenLayer.output,
			n_in =n_hidden1,
			n_out=n_hidden2,
			activation= T.tanh
		)
		self.logRegressionLayer=LogisticRegression(
			input=self.hiddenLayer2.output,
			n_in =n_hidden2,
			n_out=n_out
		)
			)





		)
		self.logRegressionLayer = LogisticRegression(
			input=self.hiddenLayer.output
			n_in=n_hidden,
			n_out=n_out
		)
		