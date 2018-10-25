import tensorflow as tf
import numpy as np
from keras.models import Sequential, Model
from keras.layers import *
from keras import backend as K
from keras.regularizers import l2
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from ICDAR2003 import ICDAR2003
import os 
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def lrn(x):
	return tf.nn.lrn(x)

def pad(x):
	padding = tf.constant([[0,0],[1,1],[1,1],[0,0]])
	return tf.pad(x,padding,'constant')

BATCH_SIZE_1 = 32
BATCH_SIZE_2 = 32
NUM_CLASSES_1 = 31
NUM_CLASSES_2 = 31
NUM_CLASSES_3 = 10
eps = 0
min_rate = .5e-16
EPOCHS = 5000
ROWS, COLS = 48,48
channels = 3

input_shape = (ROWS, COLS, channels)

intial = keras.initializers.RandomNormal(mean=0, stddev=.25,seed=random.seed(time.time()))


a = Input(shape=input_shape)
b = Conv2D(64,kernel_size=(7,7),activation='sigmoid',padding='same',data_format='channels_last',kernel_initializer=intial)(a)
c = MaxPooling2D(pool_size=(3, 3),strides=2)(b)
d = Lambda(lrn)(c)
e = Conv2D(64,kernel_size=(7,7),activation='sigmoid',padding='same',data_format='channels_last',kernel_initializer=intial)(d)
f = Lambda(lrn)(e)
g = MaxPooling2D(pool_size=(3, 3),strides=2)(f)
h = LocallyConnected2D(64,(5,5),activation='sigmoid',padding='valid',data_format='channels_last',kernel_initializer=intial)(g)
i = LocallyConnected2D(32,(5,5),activation='sigmoid',padding='valid',data_format='channels_last',kernel_initializer=intial)(h)
j = Flatten()(i)
k1 = Dense(31,activation='softmax',kernel_initializer=intial)(j)
k2 = Dense(31,activation='softmax',kernel_initializer=intial)(j)
 
model1 = Model(inputs=a, outputs=k1)
model2 = Model(inputs=a, outputs=k2)
# model3 = Model(inputs=a, outputs=k3)
learn1 = .01
learn2 = .01
optim1 = keras.optimizers.SGD(lr=learn1,decay=.0001)
optim2 = keras.optimizers.SGD(lr=learn2,decay=.0001)

model1.compile(loss=keras.losses.categorical_crossentropy,
            	optimizer=optim1,
				metrics=['accuracy'])
model2.compile(loss=keras.losses.categorical_crossentropy,
            	optimizer=optim2,
				metrics=['accuracy'])

datagen = ImageDataGenerator(
        featurewise_center=False,samplewise_center=False,featurewise_std_normalization=False,
        samplewise_std_normalization=False,zca_whitening=False,zca_epsilon=1e-06,
        rotation_range=30,width_shift_range=0.25,height_shift_range=0.25,shear_range=0.,
        zoom_range=0.25,channel_shift_range=0.0,fill_mode='nearest',cval=0.,
        horizontal_flip=False,vertical_flip=False,rescale=None,
        preprocessing_function=None,data_format=None,validation_split=0.1)

x_train_1_batches = datagen.flow_from_directory(directory='./ICDAR_reformat/1/train/',
										target_size=(48,48),
										color_mode='rgb',
										batch_size=32,
										class_mode='categorical',
										shuffle=True,
										subset='training')
x_train_2_batches = datagen.flow_from_directory(directory='./ICDAR_reformat/2/train/',
										target_size=(48,48),
										color_mode='rgb',
										batch_size=32,
										class_mode='categorical',
										shuffle=True,
										subset='training')

x_val_1_batches = datagen.flow_from_directory(directory='./ICDAR_reformat/1/train/',
										target_size=(48,48),
										color_mode='rgb',
										class_mode='categorical',
										shuffle=False,
										batch_size=1,
										subset='validation')
x_val_2_batches = datagen.flow_from_directory(directory='./ICDAR_reformat/2/train/',
										target_size=(48,48),
										color_mode='rgb',
										class_mode='categorical',
										shuffle=False,
										batch_size=1,
										subset='validation')
x_1_test = datagen.flow_from_directory(directory='./ICDAR_reformat/1/test/',
										target_size=(48,48),
										color_mode='rgb',
										class_mode='categorical',
										shuffle=False)
x_2_test = datagen.flow_from_directory(directory='./ICDAR_reformat/2/test/',
										target_size=(48,48),
										color_mode='rgb',
										class_mode='categorical',
										shuffle=False)

val1error = 0
val1acc = 0
val2error = 0
val2acc = 0

train1error = 0
train2error = 0

train1acc = 0
train2acc = 0

for ii in range(EPOCHS):

	print("Epoch {}/{}".format(ii+1,EPOCHS))
	
	train1error_sum = 0
	train2error_sum = 0

	train1acc_sum = 0
	train2acc_sum = 0

	num_batches = len(x_train_1_batches)+len(x_train_2_batches)#+len(x_train_3_batches)
	batch1_count = 0
	batch2_count = 0
	random.seed()
	


	for jj in range(num_batches): 
		train1error = 0
		train1acc = 0
		train2error = 0
		train2acc = 0
		train3error = 0

		rng = random.random()
		if rng <(len(x_train_1_batches)/num_batches) and batch1_count < len(x_train_1_batches):
			x_train_1_b,y_train_1_b = x_train_1_batches[batch1_count]
			hist1 = model1.train_on_batch(x_train_1_b, y_train_1_b)
			train1error_sum += hist1[0]
			train1acc_sum += hist1[1]
			batch1_count +=1
			# tensorboard.on_epoch_end(ii*324+jj, named_logs(model1, hist1))
		elif (rng > (len(x_train_1_batches)/num_batches)#  and rng < 1-(len(x_train_3_batches)/num_batches) 
			and batch2_count < len(x_train_2_batches)):
			x_train_2_b,y_train_2_b = x_train_2_batches[batch2_count]
			hist2 = model2.train_on_batch(x_train_2_b, y_train_2_b)
			train2error_sum += hist2[0]
			train2acc_sum += hist2[1]
			batch2_count += 1

		
		print("Batch:{:3.0f}/{}  Train1 loss: {:0.4f}  Train1 accuracy: {:0.4f}   Train2 loss: {:0.4f}  Train2 accuracy: {:0.4f}    ".
				format(jj+1,num_batches,train1error_sum/(batch1_count+.0001),train1acc_sum/(batch1_count+.0001),train2error_sum/(batch2_count+.0001),
				train2acc_sum/(batch2_count+.0001)),end='\r')

	val1error,val1acc = model1.evaluate_generator(generator=x_val_1_batches,steps=len(x_val_1_batches))
	val2error,val2acc = model2.evaluate_generator(generator=x_val_2_batches,steps=len(x_val_2_batches))

	print("Batch:{:3.0f}/{}  Train1 loss: {:0.4f}  Train1 accuracy: {:0.4f}   Train2 loss: {:0.4f}  Train2 accuracy: {:0.4f}     ".format(jj+1,num_batches,
			train1error_sum/(batch1_count+.0001),train1acc_sum/(batch1_count+.0001),train2error_sum/(batch2_count+.0001),train2acc_sum/(batch2_count+.0001)))
	print("Batch:{:3.0f}/{}  Val1 loss:   {:0.4f}  Val1 accuracy:   {:0.4f}   Val2 loss:   {:0.4f}  Val2 accuracy:   {:0.4f}\n".format(num_batches,num_batches,
			val1error,val1acc,val2error,val2acc))

	if (ii % 250 == 0):
		score = model1.evaluate_generator(generator=x_1_test,steps=len(x_1_test))
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])

		score = model2.evaluate_generator(generator=x_2_test,steps=len(x_2_test))
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])

score = model1.evaluate_generator(generator=x_1_test,steps=len(x_1_test))
print('Test loss:', score[0])
print('Test accuracy:', score[1])

score = model2.evaluate_generator(generator=x_2_test,steps=len(x_2_test))
print('Test loss:', score[0])
print('Test accuracy:', score[1])


