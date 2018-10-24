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
from keras.metrics import categorical_accuracy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
def lrn(x):
	import tensorflow as tf
	return tf.nn.lrn(x)

data = ICDAR2003('./ICDAR')

xtrain0,ytrain0,xtest0,ytest0 = data.load_data(0)
xtrain1,ytrain1,xtest1,ytest1 = data.load_data(1)


ytrain0 = keras.utils.to_categorical(ytrain0,31)
ytest0 = keras.utils.to_categorical(ytest0,31)
ytrain1 = keras.utils.to_categorical(ytrain1,31)
ytest1 = keras.utils.to_categorical(ytest1,31)

NUM_CLASSES_1 = 31
NUM_CLASSES_2 = 31

input_shape = (48,48,3)

EPOCHS = 100

xtrain0 /= 255
xtrain1 /= 255
xtest0 /= 255
ytest1 /= 255

init = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=random.seed(time.time()))

input1 = Input(shape=input_shape) 
conv1 = Conv2D(64,[7,7],padding='same',activation='sigmoid',kernel_initializer=init,input_shape=input_shape)
mp1 = MaxPooling2D([3,3],strides=2,padding='same')
lrn1 = Lambda(lrn)
conv2 = Conv2D(64,[7,7],padding='same',activation='sigmoid',kernel_initializer=init)
lrn2 = Lambda(lrn)
mp2 = MaxPooling2D([3,3],strides=2,padding='same')
lc1 = LocallyConnected2D(64,[5,5],activation='sigmoid',kernel_initializer=init)
lc2 = LocallyConnected2D(32,[5,5],activation='sigmoid',kernel_initializer=init)
flatten = Flatten()
dense0 = Dense(NUM_CLASSES_1,activation='softmax',kernel_initializer=init)
dense1 = Dense(NUM_CLASSES_1,activation='softmax',kernel_initializer=init)

SHL = flatten(lc2(lc1(mp2(lrn2(conv2(lrn1(mp1(conv1(input1)))))))))
model1 = Model(input1,dense0(SHL))
model2 = Model(input1,dense1(SHL))

model1.compile(loss=['categorical_crossentropy'],
				optimizer='SGD',
				metrics=['accuracy'])
model2.compile(loss=['categorical_crossentropy'],
				optimizer='SGD',
				metrics=['accuracy'])

x_train_1, x_val_1, y_train_1, y_val_1 = train_test_split(
	xtrain0,ytrain0,test_size=.1,random_state=random.seed(time.time()))

x_train_2, x_val_2, y_train_2, y_val_2 = train_test_split(
	xtrain1,ytrain1,test_size=.1,random_state=random.seed(time.time()))

datagen = ImageDataGenerator(
        featurewise_center=False,samplewise_center=False,featurewise_std_normalization=False,
        samplewise_std_normalization=False,zca_whitening=False,zca_epsilon=1e-06,
        rotation_range=15,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.,
        zoom_range=0.1,channel_shift_range=0.0,fill_mode='nearest',cval=0.,
        horizontal_flip=False,vertical_flip=False,rescale=None,
        preprocessing_function=None,data_format=None,validation_split=0.0)
# x_train_3_batches = datagen.flow(x_train_3,y_train_3,batch_size=BATCH_SIZE)


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

learn = .01

losses1 = []
losses2 = []
for ii in range(EPOCHS):
	if ((ii > 5 and (losses2[1]+losses1[1] - losses2[2] - losses1[2]) < 0 and learn >= -5e-11 and cooldown <= 0)  or 
		(cooldown < -100)):
		cooldown = 3
		learn = learn*np.sqrt(.1)
		print("Changing learning rate to: ",learn)
		optim = keras.optimizers.SGD(lr=learn)

		model1.compile(loss=keras.losses.categorical_crossentropy,
		            	optimizer=optim,
						metrics=[categorical_accuracy])

		model2.compile(loss=keras.losses.categorical_crossentropy,
		            	optimizer=optim,
						metrics=[categorical_accuracy])

	cooldown -= 1
	print("Epoch {}/{}".format(ii+1,EPOCHS))
	x_train_1_batches = datagen.flow(x_train_1,y_train_1,batch_size=128)
	x_train_2_batches = datagen.flow(x_train_2,y_train_2,batch_size=128)
	# x_train_3_batches = datagen.flow(x_train_3,y_train_3,batch_size=BATCH_SIZE,shuffle=True)

	train1error_sum = 0
	train2error_sum = 0

	train1acc_sum = 0
	train2acc_sum = 0

	num_batches = len(x_train_1_batches)+len(x_train_2_batches)
	batch1_count = 0
	batch2_count = 0

	random.seed(time.time())

	for jj in range(num_batches): 
		train1error = 0
		train1acc = 0
		train2error = 0
		train2acc = 0
		train3error = 0
		train3acc = 0
		rng = random.random()

		if rng < .5 and batch1_count < len(x_train_1_batches):
			x_train_1_b,y_train_1_b = x_train_1_batches[batch1_count]
			train1error,train1acc = model1.train_on_batch(x_train_1_b, y_train_1_b)
			train1error_sum += train1error
			train1acc_sum += train1acc
			batch1_count +=1

		elif (rng > .5 and batch2_count < len(x_train_2_batches)):
			x_train_2_b,y_train_2_b = x_train_2_batches[batch2_count]
			train2error,train2acc = model2.train_on_batch(x_train_2_b,y_train_2_b)
			train2error_sum += train2error
			train2acc_sum += train2acc
			batch2_count += 1

		else:
			jj -= 1

		
		print("Batch:{:3.0f}/{}  Train1 loss: {:0.4f}  Train1 accuracy: {:0.4f}   Train2 loss: {:0.4f}  Train2 accuracy: {:0.4f}    ".
				format(jj+1,num_batches,train1error_sum/(batch1_count+.0001),train1acc_sum/(batch1_count+.0001),train2error_sum/(batch2_count+.0001),
				train2acc_sum/(batch2_count+.0001)),end='\r')
	# import pdb
	val1error,val1acc = model1.test_on_batch(x_val_1,y_val_1)
	val2error,val2acc = model2.test_on_batch(x_val_2,y_val_2)
	# val3error,val3acc = model3.test_on_batch(x_val_3,y_val_3)
	
	train1error = train1error_sum/batch1_count
	losses1 += [train1error]
	if (len(losses1) > 3):
		losses1.pop(0)
	train1acc = train1acc_sum/batch1_count
	
	train2error = train2error_sum/batch2_count
	losses2 += [train2error]
	if (len(losses2) > 3):
		losses2.pop(0)
	train2acc = train2acc_sum/batch2_count

	print("Batch:{:3.0f}/{}  Train1 loss: {:0.4f}  Train1 accuracy: {:0.4f}   Train2 loss: {:0.4f}  Train2 accuracy: {:0.4f}     ".format(jj+1,num_batches,
			train1error_sum/(batch1_count+.0001),train1acc_sum/(batch1_count+.0001),train2error_sum/(batch2_count+.0001),train2acc_sum/(batch2_count+.0001)))
	print("Batch:{:3.0f}/{}  Val1 loss:   {:0.4f}  Val1 accuracy:   {:0.4f}   Val2 loss:   {:0.4f}  Val2 accuracy:   {:0.4f}\n".format(num_batches,num_batches,
			val1error,val1acc,val2error,val2acc))
	# print("Train2 loss: ",train2error, " Train2 accuracy: ", train2acc, " Val2 loss: ", val2error, " Val2 accuracy: ", val2acc)

score = model1.test_on_batch(xtest0, ytest0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
score = model2.test_on_batch(xtest1, ytest1)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


