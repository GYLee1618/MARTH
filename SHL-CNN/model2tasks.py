import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from ICDAR2003 import ICDAR2003
import os 
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tqdm import tqdm
import random
import time
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def lrn(x):
	import tensorflow as tf
	return tf.nn.lrn(x)

def pad(x):
	padding = tf.constant([[0,0],[2,2],[2,2],[0,0]])
	return tf.pad(x,padding,'constant')

BATCH_SIZE_1 = 32
#BATCH_SIZE_2 = 16
NUM_CLASSES_1 = 31
NUM_CLASSES_2 = 31
NUM_CLASSES_3 = 10
EPOCHS = 10000
eps = 0
min_rate = .5e-16

ROWS, COLS = 48,48
channels = 3


icdar2003 = ICDAR2003('./ICDAR')

x_train_1,y_train_1,x_test_1,y_test_1 = icdar2003.load_data(0)

input_shape = (ROWS, COLS, channels)

x_train_1 = x_train_1.astype('float32')
# x_test_1 = x_test_1.astype('float32')
x_train_1 /= 255
# x_test_1 /= 255



y_train_1 = keras.utils.to_categorical(y_train_1, NUM_CLASSES_1)
y_test_1 = keras.utils.to_categorical(y_test_1, NUM_CLASSES_1)

x_train_1, x_val_1, y_train_1, y_val_1 = train_test_split(
	x_train_1,y_train_1,test_size=.1,random_state=random.seed(time.time()))


intial = keras.initializers.RandomNormal(mean=0, stddev=.01,seed=random.seed(time.time()))


a = Input(shape=input_shape)
b = Conv2D(64,kernel_size=(7,7),activation='relu',padding='same',data_format='channels_last',kernel_initializer=intial)(a)
c = MaxPooling2D(pool_size=(3, 3),strides=2)(b)
l = Activation('linear')(c)
d = Lambda(lrn)(l)
e = Conv2D(64,kernel_size=(7,7),activation='relu',padding='same',data_format='channels_last',kernel_initializer=intial)(d)
f = Lambda(lrn)(e)
g = MaxPooling2D(pool_size=(3, 3),strides=2)(f)
ll = Activation('linear')(g)
#p = Lambda(pad)(g)
h = LocallyConnected2D(64,(5,5),activation='relu',padding='valid',data_format='channels_last',kernel_initializer=intial)(ll)
#pp = Lambda(pad)(h)
i = LocallyConnected2D(32,(5,5),activation='relu',padding='valid',data_format='channels_last',kernel_initializer=intial)(h)
j = Flatten()(i)
k1 = Dense(NUM_CLASSES_1,activation='softmax',kernel_initializer=intial)(j)

model1 = Model(inputs=a, outputs=k1)

learn1 = .01

optim1 = keras.optimizers.SGD(lr=learn1,decay=.001)


model1.compile(loss=keras.losses.categorical_crossentropy,
            	optimizer=optim1,
				metrics=['accuracy'])
model1.summary()
#model1 = load_model('SHL-CNN1.h5')


datagen = ImageDataGenerator(
        featurewise_center=False,samplewise_center=False,featurewise_std_normalization=False,
        samplewise_std_normalization=False,zca_whitening=False,zca_epsilon=1e-06,
        rotation_range=15,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.,
        zoom_range=0.1,channel_shift_range=0.0,fill_mode='nearest',cval=0.,
        horizontal_flip=False,vertical_flip=False,rescale=None,
        preprocessing_function=None,data_format=None,validation_split=0.0)

#x_train_1_batches = datagen.flow(x_train_1,y_train_1,batch_size=BATCH_SIZE_1)
datagen.fit(x_train_1)


model.fit_generator(datagen.flow(x_train_1, y_train_1,batch_size=BATCH_SIZE),
          epochs=EPOCHS,
          steps_per_epoch=len(x_train_1)/BATCH_SIZE,
          verbose=1,
          validation_data=(x_val_1,y_val_1))


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



# val1error = 0
# val1acc = 0
# train1error = 0
# train1acc = 0
# lastloss1 = 0
# cooldown = 0

# # (len(x_train_1_batches)/num_batches)
# losses1 = []
# losses2 = []
# losses3 = []

# for ii in range(EPOCHS):

# 	if ii % 100 == 0:
# 	 	model1.save('SHL-CNN1.h5')
# 	try:
# 		print(losses1[0] - losses1[2])
# 	except:
# 		pass

# 	if ((ii > 5 and (losses1[0] - losses1[2]) < eps and learn1 >= min_rate and cooldown <= 0)  or 
# 		(cooldown < -100)):
# 		cooldown = 3
# 		learn1 = learn1*np.sqrt(.1)
# 		print("Changing learning rate to: ",learn1)#,learn2)
# 		optim1 = keras.optimizers.SGD(lr=learn1)

# 		model1.compile(loss=keras.losses.categorical_crossentropy,
# 		            	optimizer=optim1,
# 						metrics=['accuracy'])

# 	cooldown -= 1
# 	print("Epoch {}/{}".format(ii+1,EPOCHS))
# 	x_train_1_batches = datagen.flow(x_train_1,y_train_1,batch_size=BATCH_SIZE_1,shuffle=True)
# 	train1error_sum = 0
# 	train1acc_sum = 0
# 	num_batches = len(x_train_1_batches)#+len(x_train_2_batches)#+len(x_train_3_batches)
# 	batch1_count = 0
	
	

# 	for jj in range(num_batches): 
# 		train1error = 0
# 		train1acc = 0
# 		x_train_1_b,y_train_1_b = x_train_1_batches[batch1_count]
# 		hist1 = model1.fit(x_train_1_b, y_train_1_b,batch_size=x_train_1_b.shape[0],verbose=0)
# 		train1error_sum += hist1.history['loss'][0]
# 		train1acc_sum += hist1.history['acc'][0]
# 		batch1_count +=1

		
# 		print("Batch:{:3.0f}/{}  Train1 loss: {:0.4f}  Train1 accuracy: {:0.4f}     ".
# 				format(jj+1,num_batches,train1error_sum/(batch1_count+.0001),train1acc_sum/(batch1_count+.0001)),end='\r')

# 	index1 =int(np.floor(x_val_1.shape[0]/2))
# 	val1error1,val1acc1 = model1.test_on_batch(x_val_1[0:index1,:],y_val_1[0:index1,:])
# 	val1error2,val1acc2 = model1.test_on_batch(x_val_1[index1:,:],y_val_1[index1:,:])
# 	val1error = (val1error1+val1error2)/2
# 	val1acc = (val1acc1+val1acc2)/2
	
# 	train1error = train1error_sum/batch1_count
# 	losses1 += [train1error]
# 	if (len(losses1) > 3):
# 		losses1.pop(0)
# 	train1acc = train1acc_sum/batch1_count
	

# 	print("Batch:{:3.0f}/{}  Train1 loss: {:0.4f}  Train1 accuracy: {:0.4f}     ".format(jj+1,num_batches,
# 			train1error_sum/(batch1_count+.0001),train1acc_sum/(batch1_count+.0001)))
# 	print("Batch:{:3.0f}/{}  Val1 loss:   {:0.4f}  Val1 accuracy:   {:0.4f}\n".format(num_batches,num_batches,
# 			val1error,val1acc))

# from keras.models import load_model

# model.save('SHL-CNN.h5')


'''
layer1 = model1.get_layer(index = 7)
layer2 = model2.get_layer(index = 7)


if layer1 == layer2:
	print("BOOM2")





score = model2.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''




