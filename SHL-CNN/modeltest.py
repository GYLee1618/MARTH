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
from tqdm import tqdm
import random
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def lrn(x):
	return tf.nn.lrn(x)

BATCH_SIZE = 64
NUM_CLASSES_1 = 26
NUM_CLASSES_2 = 36
EPOCHS = 10000
eps = .0005
min_rate = .5e-07

ROWS, COLS = 48,48
channels = 3

input_shape = (ROWS, COLS, 3)

icdar2003 = ICDAR2003('./ICDAR')

x_train_1,y_train_1,x_test_1,y_test_1 = icdar2003.load_data(0)
x_train_2,y_train_2,x_test_2,y_test_2 = icdar2003.load_data(1)

# x_train_1 = x_train_1.reshape(x_train_1.shape[0], ROWS, COLS, channels)
# x_test_1 = x_test_1.reshape(x_test_1.shape[0], ROWS, COLS, channels)

# x_train_2 = x_train_2.reshape(x_train_2.shape[0], ROWS, COLS, channels)
# x_test_2 = x_test_2.reshape(x_test_2.shape[0], ROWS, COLS, channels)

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
	x_train_1,y_train_1,test_size=.1,random_state=random.seed(time.time()))

x_train_2, x_val_2, y_train_2, y_val_2 = train_test_split(
	x_train_2,y_train_2,test_size=.1,random_state=random.seed(time.time()))

intial = keras.initializers.RandomNormal(mean=0, stddev=.25,seed=random.seed(time.time()))


a = Input(shape=input_shape)
b = Conv2D(64, kernel_size=(2, 2),activation='relu',padding='same')(a)
c = BatchNormalization()(b)
d = Activation('relu')(c)
e = Dropout(0.3)(d)
f = Conv2D(64,kernel_size=(4,4),activation='relu',padding='same')(e)
g = BatchNormalization()(f)
h = Activation('relu')(g)
i = MaxPooling2D(pool_size=(2, 2),strides=2)(h)
j = Flatten()(i)
k1 = Dense(NUM_CLASSES_1,activation='softmax',kernel_initializer=intial)(j)
k2 = Dense(NUM_CLASSES_2,activation='softmax',kernel_initializer=intial)(j)

# a = Input(shape=input_shape)
# b = Conv2D(64,kernel_size=(7,7),activation='sigmoid',padding='same',data_format='channels_last',kernel_initializer=intial)(a)
# c = MaxPooling2D(pool_size=(3, 3),strides=2)(b)
# d = Lambda(lrn)(c)
# e = Conv2D(64,kernel_size=(7,7),activation='sigmoid',padding='same',data_format='channels_last',kernel_initializer=intial)(d)
# f = Lambda(lrn)(e)
# g = MaxPooling2D(pool_size=(3, 3),strides=2)(f)
# h = LocallyConnected2D(64,(5,5),activation='sigmoid',padding='valid',data_format='channels_last',kernel_initializer=intial)(g)
# i = LocallyConnected2D(32,(5,5),activation='sigmoid',padding='valid',data_format='channels_last',kernel_initializer=intial)(h)
# j = Flatten()(i)
# k1 = Dense(NUM_CLASSES_1,activation='softmax',kernel_initializer=intial)(j)
# k2 = Dense(NUM_CLASSES_2,activation='softmax',kernel_initializer=intial)(j)
 
model1 = Model(inputs=a, outputs=k1)
model2 = Model(inputs=a, outputs=k2)
learn = .01
optim = keras.optimizers.SGD(lr=learn, momentum=.5)

model1.compile(loss=keras.losses.categorical_crossentropy,
            	optimizer=optim,
				metrics=['accuracy'])
model2.compile(loss=keras.losses.categorical_crossentropy,
            	optimizer=optim,
				metrics=['accuracy'])

# layer1 = model1.get_layer(index = 7)
# layer2 = model2.get_layer(index = 7)




# if layer1 == layer2:
# 	print("BOOM")

datagen = ImageDataGenerator(
        featurewise_center=False,samplewise_center=False,featurewise_std_normalization=False,
        samplewise_std_normalization=False,zca_whitening=False,zca_epsilon=1e-06,
        rotation_range=15,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.,
        zoom_range=0.1,channel_shift_range=0.0,fill_mode='nearest',cval=0.,
        horizontal_flip=False,vertical_flip=False,rescale=None,
        preprocessing_function=None,data_format=None,validation_split=0.0)

x_train_1_batches = datagen.flow(x_train_1,y_train_1,batch_size=BATCH_SIZE)
x_train_2_batches = datagen.flow(x_train_2,y_train_2,batch_size=BATCH_SIZE)


val1error = 0
val1acc = 0
val2error = 0
val2acc = 0
train1error = 0
train2error = 0
train1acc = 0
train2acc = 0
lastloss1 = 0
lastloss2 = 0
cooldown = 0
from keras.models import load_model

losses1 = []
losses2 = []

for ii in range(EPOCHS):
	
	if ii % 100 == 0:
		model1.save('SHL-CNN1.h5')
		model2.save('SHL-CNN2.h5')
	# total_loss = [(-(losses1[i+1]+losses2[i+1]) + (losses1[i]+losses2[i])) for i in range(len(losses1)-1)]
	# print(total_loss)
	try:
		print(losses2[2]+losses1[2] - losses2[4] - losses1[4])
	except:
		pass

	if ((ii > 5 and (losses2[2]+losses1[2] - losses2[4] - losses1[4]) < eps and learn >= min_rate and cooldown <= 0)  or 
		(cooldown < -100)):
		cooldown = 5
		learn = learn*.5
		print("Changing learning rate to: ",learn)
		optim = keras.optimizers.SGD(lr=learn,momentum=.5)

		model1.compile(loss=keras.losses.categorical_crossentropy,
		            	optimizer=optim,
						metrics=['accuracy'])

		model2.compile(loss=keras.losses.categorical_crossentropy,
		            	optimizer=optim,
						metrics=['accuracy'])
	cooldown -= 1
	print("Epoch {}/{}".format(ii+1,EPOCHS))
	x_train_1_batches = datagen.flow(x_train_1,y_train_1,batch_size=BATCH_SIZE,shuffle=True)
	x_train_2_batches = datagen.flow(x_train_2,y_train_2,batch_size=BATCH_SIZE,shuffle=True)
	train1error_sum = 0
	train2error_sum = 0
	train1acc_sum = 0
	train2acc_sum = 0
	num_batches = len(x_train_1_batches)+len(x_train_2_batches)
	batch1_count = 0
	batch2_count = 0
	random.seed()
	
	

	for jj in range(num_batches): 
		train1error = 0
		train1acc = 0
		train2error = 0
		train2acc = 0
		
		if random.random() >.5:
			if batch1_count >= len(x_train_1_batches):
				x_train_2_b,y_train_2_b = x_train_2_batches[batch2_count]
				train2error,train2acc = model2.train_on_batch(x_train_2_b,y_train_2_b)
				train2error_sum += train2error
				train2acc_sum += train2acc
				batch2_count += 1
			else:
				x_train_1_b,y_train_1_b = x_train_1_batches[batch1_count]
				train1error,train1acc = model1.train_on_batch(x_train_1_b, y_train_1_b)
				train1error_sum += train1error
				train1acc_sum += train1acc
				batch1_count +=1
			#train2error,train2acc = model2.train_on_batch(x_train_2_b,y_train_2_b)
		else:
			if batch2_count >= len(x_train_2_batches):
				x_train_1_b,y_train_1_b = x_train_1_batches[batch1_count]
				train1error,train1acc = model1.train_on_batch(x_train_1_b, y_train_1_b)
				train1error_sum += train1error
				train1acc_sum += train1acc
				batch1_count +=1
			else:
				x_train_2_b,y_train_2_b = x_train_2_batches[batch2_count]
				train2error,train2acc = model2.train_on_batch(x_train_2_b,y_train_2_b)
				train2error_sum += train2error
				train2acc_sum += train2acc
				batch2_count += 1
			#train1error,train1acc = model1.train_on_batch(x_train_1_b, y_train_1_b)
		# train1error_sum += train1error
		# train1acc_sum += train1acc
		# train2error_sum += train2error
		# train2acc_sum += train2acc
		
		print("Batch:{:3.0f}/{}  Train1 loss: {:0.4f}  Train1 accuracy: {:0.4f}   Train2 loss: {:0.4f}  Train2 accuracy: {:0.4f}     ".format(jj+1,num_batches,
			train1error_sum/(batch1_count+.0001),train1acc_sum/(batch1_count+.0001),train2error_sum/(batch2_count+.0001),train2acc_sum/(batch2_count+.0001)),end='\r')

	val1error,val1acc = model1.test_on_batch(x_val_1,y_val_1)
	val2error,val2acc = model2.test_on_batch(x_val_2,y_val_2)
	train1error = train1error_sum/num_batches
	losses1 += [train1error]
	if (len(losses1) > 5):
		losses1.pop(0)
	train1acc = train1acc_sum/num_batches
	train2error = train2error_sum/num_batches
	losses2 += [train2error]
	if (len(losses2) > 5):
		losses2.pop(0)
	train2acc = train2acc_sum/num_batches

	print("Batch:{:3.0f}/{}  Train1 loss: {:0.4f}  Train1 accuracy: {:0.4f}   Train2 loss: {:0.4f}  Train2 accuracy: {:0.4f}     ".format(jj+1,num_batches,
			train1error_sum/(batch1_count+.0001),train1acc_sum/(batch1_count+.0001),train2error_sum/(batch2_count+.0001),train2acc_sum/(batch2_count+.0001)))
	print("Batch:{:3.0f}/{}  Val1 loss:   {:0.4f}  Val1 accuracy:   {:0.4f}   Val2 loss:   {:0.4f}  Val2 accuracy:   {:0.4f}\n".format(num_batches,num_batches,
			val1error,val1acc,val2error,val2acc))
	# print("Train2 loss: ",train2error, " Train2 accuracy: ", train2acc, " Val2 loss: ", val2error, " Val2 accuracy: ", val2acc)


from keras.models import load_model

model.save('SHL-CNN.h5')


'''
layer1 = model1.get_layer(index = 7)
layer2 = model2.get_layer(index = 7)


if layer1 == layer2:
	print("BOOM2")





score = model2.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''




