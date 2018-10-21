import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras import backend as K
from keras.regularizers import l2
import keras
from ICDAR2003 import ICDAR2003
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

BATCH_SIZE = 128
NUM_CLASSES_EN = 62
NUM_CLASSES_RUSS = 500
EPOCHS = 100

ROWS, COLS = 48,48

input_shape = (ROWS, COLS, 3)


def SHL(input_shape):
	model = Sequential()
	model.add(Conv2D(64, kernel_size=(9, 9),activation='sigmoid',input_shape=input_shape,padding='same'))
	model.add(MaxPooling2D(pool_size=(3, 3),strides=2))
	model.add(BatchNormalization())
	model.add(Conv2D(64, kernel_size=(9, 9),activation='sigmoid',padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(3, 3),strides=2))
	model.add(LocallyConnected2D(64,kernel_size=(6,6),activation='sigmoid'))
	model.add(LocallyConnected2D(32,kernel_size=(6,6),activation='sigmoid'))
	return model

def output_layer(model,nodes):
	model.add(Dense(nodes,activation='softmax'))
	return model

icdar2003 = ICDAR2003('./ICDAR',NUM_CLASSES_EN)
x_train,y_train,x_test,y_test = icdar2003.load_data()
x_train = x_train/255.

# model = SHL(input_shape)
# model = output_layer(model,NUM_CLASSES_EN)
model = Sequential()
model.add(Conv2D(64, kernel_size=(9, 9),activation='sigmoid',input_shape=input_shape,padding='same',data_format='channels_last',kernel_regularizer=l2(0.01)))
model.add(MaxPooling2D(pool_size=(3, 3),strides=2))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(9, 9),activation='sigmoid',padding='same',kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3),strides=2))
# model.add(Flatten())
model.add(LocallyConnected2D(64,(5,5),activation='sigmoid',padding='valid',data_format='channels_last'))
model.add(LocallyConnected2D(32,(5,5),activation='sigmoid',padding='valid',data_format='channels_last'))
model.add(Flatten())
model.add(Dense(NUM_CLASSES_EN,activation='softmax',kernel_regularizer=l2(0.01)))

model.compile(loss=keras.losses.categorical_crossentropy,
				optimizer=keras.optimizers.Adam(),
				metrics=['accuracy'])

# import pdb
# pdb.set_trace()

model.fit(x_train, y_train,
			batch_size=BATCH_SIZE,
			epochs=EPOCHS,
			verbose=1,
			validation_split=.2)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
