import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,normalization,LocallyConnected2D
from keras import backend as K
from sklearn.model_selection import train_test_split


BATCH_SIZE = 128
NUM_CLASSES_EN = 62
NUM_CLASSES_RUSS = 500
EPOCHS = 3

ROWS, COLS = 48,48

input_shape = (ROWS, COLS, 3)


def SHL(input_shape):
	model = Sequential()
	model.add(Conv2D(64, kernel_size=(9, 9),activation='sigmoid',input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(3, 3),strides=2))
	model.add(normalization.LRN2D(n=9))
	model.add(Conv2D(64, kernel_size=(9, 9),activation='sigmoid'))
	model.add(normalization.LRN2D(n=9))
	model.add(MaxPooling2D(pool_size=(3, 3),strides=2))
	model.add(LocallyConnected2D(64,kernal_size=(7,7),activation='sigmoid'))
	model.add(LocallyConnected2D(32,kernal_size=(7,7),activation='sigmoid'))
    return model

def output_layer(model,nodes):
    model.add(Dense(nodes,activation='softmax'))
    return model

model = SHL(input_shape)
model = output_layer(model,62)

model.compile(loss=keras.losses.categorical_crossentropy,
            	optimizer=keras.optimizers.Adam(),
				metrics=['accuracy'])


