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
from keras.datasets import cifar100
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def lrn(x):
	return tf.nn.lrn(x)

BATCH_SIZE = 64
NUM_CLASSES_1 = 31
NUM_CLASSES_2 = 31
EPOCHS = 1

ROWS, COLS = 48,48
channels = 3

input_shape = (ROWS, COLS, 3)

icdar2003 = ICDAR2003('./ICDAR')

x_train_1,y_train_1,x_test_1,y_test_1 = icdar2003.load_data(0)
x_train_2,y_train_2,x_test_2,y_test_2 = icdar2003.load_data(1)

x_train_1 = x_train_1.reshape(x_train_1.shape[0], ROWS, COLS, channels)
x_test_1 = x_test_1.reshape(x_test_1.shape[0], ROWS, COLS, channels)

x_train_2 = x_train_2.reshape(x_train_2.shape[0], ROWS, COLS, channels)
x_test_2 = x_test_2.reshape(x_test_2.shape[0], ROWS, COLS, channels)

input_shape = (ROWS, COLS, channels)

x_train_1 = x_train_1.astype('float32')
x_test_1 = x_test_1.astype('float32')
x_train_1 /= 255
x_test_1 /= 255


x_train_2 = x_train_2.astype('float32')
x_test_2 = x_test_2.astype('float32')
x_train_2 /= 255
x_test_2 /= 255



# y_train_1 = keras.utils.to_categorical(y_train_1, NUM_CLASSES_EN)
# y_test_1 = keras.utils.to_categorical(y_test_1, NUM_CLASSES_EN)
# y_train_2 = keras.utils.to_categorical(y_train_2, NUM_CLASSES_EN)
# y_test_2 = keras.utils.to_categorical(y_test_2, NUM_CLASSES_EN)

x_train_1, x_val_1, y_train_1, y_val_1 = train_test_split(
	x_train_1,y_train_1,test_size=.1)

x_train_2, x_val_2, y_train_2, y_val_2 = train_test_split(
	x_train_2,y_train_2,test_size=.1)



a = Input(shape=input_shape)
b = Conv2D(32,kernel_size=(7,7),activation='sigmoid',padding='same',data_format='channels_last',kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(a)
c = MaxPooling2D(pool_size=(3, 3),strides=2)(b)
d = Lambda(lrn)(c)
e = Conv2D(32,kernel_size=(7,7),activation='sigmoid',padding='same',data_format='channels_last',kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(d)
f = MaxPooling2D(pool_size=(3, 3),strides=2)(e)
g = Lambda(lrn)(f)
h = LocallyConnected2D(64,(5,5),activation='sigmoid',padding='valid',data_format='channels_last',kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(g)
i = LocallyConnected2D(32,(5,5),activation='sigmoid',padding='valid',data_format='channels_last',kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(h)
j = Flatten()(i)
k1 = Dense(NUM_CLASSES_1,activation='softmax',kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(j)
k2 = Dense(NUM_CLASSES_2,activation='softmax',kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(j)
 
model1 = Model(inputs=a, outputs=k1)
model2 = Model(inputs=a, outputs=k2)


model1.compile(loss=keras.losses.categorical_crossentropy,
            	optimizer=keras.optimizers.Adam(lr=.0005),
				metrics=['accuracy'])
model2.compile(loss=keras.losses.categorical_crossentropy,
            	optimizer=keras.optimizers.Adam(lr=.0005),
				metrics=['accuracy'])

layer1 = model1.get_layer(index = 7)
layer2 = model2.get_layer(index = 7)




if layer1 == layer2:
	print("BOOM")

datagen = ImageDataGenerator(
        featurewise_center=False,samplewise_center=False,featurewise_std_normalization=False,
        samplewise_std_normalization=False,zca_whitening=False,zca_epsilon=1e-06,
        rotation_range=60,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.,
        zoom_range=0.2,channel_shift_range=0.,fill_mode='nearest',cval=0.,
        horizontal_flip=False,vertical_flip=False,rescale=None,
        preprocessing_function=None,data_format=None,validation_split=0.0)

x_train_1_batches = datagen.flow(x_train_1,y_train_1,batch_size=BATCH_SIZE)
x_train_2_batches = datagen.flow(x_train_2,y_train_2,batch_size=BATCH_SIZE)

m_m,n_n = x_train_1_batches[0]

val1error = 0
val1acc = 0
val2error = 0
val2acc = 0
train1error = 0
train2error = 0
train1acc = 0
train2acc = 0


for ii in range(EPOCHS):
	print("Epoch: ", ii)
	x_train_1_batches = datagen.flow(x_train_1,y_train_1,batch_size=BATCH_SIZE,shuffle=True)
	x_train_2_batches = datagen.flow(x_train_2,y_train_2,batch_size=BATCH_SIZE,shuffle=True)
	x_train_1_b,y_train_1_b = next(x_train_1_batches)
	x_train_2_b,y_train_2_b = next(x_train_2_batches)
	train1error_sum = 0
	train2error_sum = 0
	train1acc_sum = 0
	train2acc_sum = 0
	num_batches = min(len(x_train_1_batches),len(x_train_2_batches))
	for jj in tqdm(range(num_batches)): 
		x_train_1_b,y_train_1_b = x_train_1_batches[jj]
		x_train_2_b,y_train_2_b = x_train_2_batches[jj]
		train1error,train1acc = model1.train_on_batch(x_train_1_b, y_train_1_b)
		train2error,train2acc = model2.train_on_batch(x_train_2_b,y_train_2_b)
		train1error_sum += train1error
		train1acc_sum += train1acc
		train2error_sum += train2error
		train2acc_sum += train2acc
	val1error,val1acc = model1.test_on_batch(x_val_1,y_val_1)
	val2error,val2acc = model2.test_on_batch(x_val_2,y_val_2)
	train1error = train1error_sum/num_batches
	train1acc = train1acc_sum/num_batches
	train2error = train2error_sum/num_batches
	train2acc = train2acc_sum/num_batches
	print("Train1 loss: ",train1error, " Train1 accuracy: ", train1acc, " Val1 loss: ", val1error, " Val1 accuracy: ", val1acc)
	print("Train2 loss: ",train2error, " Train2 accuracy: ", train2acc, " Val2 loss: ", val2error, " Val2 accuracy: ", val2acc)



'''
layer1 = model1.get_layer(index = 7)
layer2 = model2.get_layer(index = 7)


if layer1 == layer2:
	print("BOOM2")





score = model2.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''




