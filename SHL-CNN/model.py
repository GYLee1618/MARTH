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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def lrn(x):
	return tf.nn.lrn(x)

BATCH_SIZE = 64
NUM_CLASSES_EN = 62
NUM_CLASSES_RUSS = 500
EPOCHS = 1000

ROWS, COLS = 48,48

input_shape = (ROWS, COLS, 3)

icdar2003 = ICDAR2003('./ICDAR',NUM_CLASSES_EN)
x_train,y_train,x_test,y_test = icdar2003.load_data()
x_train = x_train/255.
x_test = x_test/255.

x_train, x_val, y_train, y_val = train_test_split(
    x_train,y_train,test_size=.1,random_state=None)

# model = SHL(input_shape)
# model = output_layer(model,NUM_CLASSES_EN)
model = Sequential()
model.add(Conv2D(64, kernel_size=(7, 7),activation='sigmoid',input_shape=input_shape,padding='same',
				data_format='channels_last',
				kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01)))
model.add(MaxPooling2D(pool_size=(3, 3),strides=2))
model.add(Lambda(lrn))
model.add(Conv2D(64, kernel_size=(7, 7),activation='sigmoid',padding='same',
				kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01)))
# model.add(BatchNormalization())
model.add(Lambda(lrn))
model.add(MaxPooling2D(pool_size=(3, 3),strides=2))
# model.add(Flatten())
model.add(LocallyConnected2D(64,(5,5),activation='sigmoid',padding='valid',data_format='channels_last',
				kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01)))
# model.add(Dropout(.2))
model.add(LocallyConnected2D(32,(5,5),activation='sigmoid',padding='valid',data_format='channels_last',
				kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01)))
# model.add(Dropout(.2))
model.add(Flatten())
model.add(Dense(NUM_CLASSES_EN,activation='softmax',
				kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01)))

model.compile(loss=keras.losses.categorical_crossentropy,
				optimizer=keras.optimizers.SGD(lr=.001),
				metrics=['accuracy'])

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(.01),
                               cooldown=0,
                               patience=5,
                               min_lr=.5e-10,
                               verbose=1)

callbacks = [lr_reducer]

# import pdb
# pdb.set_trace()
datagen = ImageDataGenerator(
        featurewise_center=False,samplewise_center=False,featurewise_std_normalization=False,
        samplewise_std_normalization=False,zca_whitening=False,zca_epsilon=1e-06,
        rotation_range=60,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.,
        zoom_range=0.2,channel_shift_range=0.,fill_mode='nearest',cval=0.,
        horizontal_flip=False,vertical_flip=False,rescale=None,
        preprocessing_function=None,data_format=None,validation_split=0.0)

datagen.fit(x_train)


model.fit_generator(datagen.flow(x_train, y_train,batch_size=BATCH_SIZE),
          epochs=EPOCHS,
          callbacks=callbacks,
          steps_per_epoch=len(x_train)/BATCH_SIZE,
          verbose=1,
          validation_data=(x_val,y_val))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

from keras.models import load_model

model.save('SHL-CNN.h5')