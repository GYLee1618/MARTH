import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras import backend as K
from keras.regularizers import l2
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from ICDAR2003 import ICDAR2003
import os 
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.datasets import cifar10
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def lrn(x):
	return tf.nn.lrn(x)

BATCH_SIZE = 64
NUM_CLASSES = 10
NUM_CLASSES_EN = 62
NUM_CLASSES_RUSS = 500
EPOCHS = 1000

ROWS, COLS = 32,32
channels = 3

input_shape = (ROWS, COLS, 3)

(x_train,y_train),(x_test,y_test) = cifar10.load_data()

x_train = x_train.reshape(x_train.shape[0], ROWS, COLS, channels)
x_test = x_test.reshape(x_test.shape[0], ROWS, COLS, channels)

input_shape = (ROWS, COLS, channels)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)



a = Input(shape=input_shape)
b = Conv2D(32,kernel_size=(7,7),activation='sigmoid',padding='same',data_format='channels_last',kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(a)
c = MaxPooling2D(pool_size=(3, 3),strides=2)(b)
d = Lambda(lrn)(c)
e = Conv2D(32,kernel_size=(7,7),activation='sigmoid',padding='same',data_format='channels_last',kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(d)
f = MaxPooling2D(pool_size=(3, 3),strides=2)(e)
g = Lambda(lrn)(f)
h = LocallyConnected2D(64,(3,3),activation='sigmoid',padding='valid',data_format='channels_last',kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(g)
i = LocallyConnected2D(32,(3,3),activation='sigmoid',padding='valid',data_format='channels_last',kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(h)
j = Flatten()(i)
k1 = Dense(NUM_CLASSES,activation='softmax',kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(j)
k2 = Dense(100,activation='softmax',kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(j)
 
model1 = Model(inputs=a, outputs=k1)
model2 = Model(inputs=a, outputs=k2)


model1.compile(loss=keras.losses.categorical_crossentropy,
            	optimizer=keras.optimizers.Adam(),
				metrics=['accuracy'])

model1.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,
          validation_split=.1)

score = model2.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])




