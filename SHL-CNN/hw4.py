#Isaac Alboucai
#CGML HW4
#Professor Curro
#October 3, 2018
#Reference https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
import ICDAR2003
import numpy as np
import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


BATCH_SIZE = 32
NUM_CLASSES = 31
EPOCHS = 200

def lr_schedule(epoch):
    lr = 1e-3
    
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


rows, cols,channels = 48,48,3
data = ICDAR2003.ICDAR2003('./ICDAR')
x_train,y_train,x_test,y_test = data.load_data(1)

x_train, x_val, y_train, y_val = train_test_split(
	x_train,y_train,test_size=.1,random_state=2345432)


x_train = x_train.reshape(x_train.shape[0], rows, cols, channels)
x_test = x_test.reshape(x_test.shape[0], rows, cols, channels)
x_val = x_val.reshape(x_val.shape[0],rows,cols,channels)

input_shape = (rows, cols, channels)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_val = x_val.astype('float32')
x_train /= 255
x_test /= 255
x_val /=255

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
y_val = keras.utils.to_categorical(y_val, NUM_CLASSES)

model = Sequential()


model.add(Conv2D(32, kernel_size=(2, 2),activation='relu',padding='same',input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Conv2D(32,kernel_size=(4,4),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))

#model.add(Conv2D(48, kernel_size=(3, 3),activation='relu',padding='same'))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(Conv2D(48, kernel_size=(5, 5),activation='relu',padding='same'))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(3, 3),strides=2))

model.add(Conv2D(64, kernel_size=(4, 4),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(.4))
model.add(Conv2D(64, kernel_size=(2, 2),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))

model.add(Conv2D(128, kernel_size=(2, 2),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(.25))
model.add(Conv2D(128, kernel_size=(3, 3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))

model.add(Flatten())
#model.add(Dense(48, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01)))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
            	optimizer=keras.optimizers.Adam(lr=lr_schedule(0)),
				metrics=['accuracy'])

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [lr_reducer,lr_scheduler]

datagen = ImageDataGenerator(
        featurewise_center=False,samplewise_center=False,featurewise_std_normalization=False,
        samplewise_std_normalization=False,zca_whitening=False,zca_epsilon=1e-06,
        rotation_range=0,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.,
        zoom_range=0.,channel_shift_range=0.,fill_mode='nearest',cval=0.,
        horizontal_flip=True,vertical_flip=False,rescale=None,
        preprocessing_function=None,data_format=None,validation_split=0.0)

datagen.fit(x_train)
model.summary()
model.fit_generator(datagen.flow(x_train, y_train,batch_size=BATCH_SIZE),
          epochs=EPOCHS,
          steps_per_epoch=len(x_train)/BATCH_SIZE,
          verbose=1,
          callbacks=callbacks,
          validation_data=(x_val, y_val))

score = model.test_on_batch(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

