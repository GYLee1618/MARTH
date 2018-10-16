import tensorflow as tf
from tensorflow.keras import *

def init_model():
    model = Sequential()

def conv_layers(model,trainable):
    raise NotImplementedError

def output_layer(model,nodes):
    model.add(Dense(nodes,activation='softmax'))

