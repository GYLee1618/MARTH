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

def lr_schedule(epoch):
	lr = 1e-2
	if epoch > 200:
		lr *= 0.5e-3
	elif epoch > 150:
		lr *= 1e-3
	elif epoch > 100:
		lr *= 1e-2
	elif epoch > 40:
		lr *= 1e-1
	print('Learning rate: ', lr)
	return lr 

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
EPOCHS = 1
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


intial = keras.initializers.RandomNormal(mean=0, stddev=.25,seed=random.seed(time.time()))


a = Input(shape=input_shape)
b = Conv2D(64,kernel_size=(7,7),activation='sigmoid',padding='same',data_format='channels_last',kernel_initializer=intial)(a)
c = MaxPooling2D(pool_size=(3, 3),strides=2)(b)
l = Activation('linear')(c)
d = Lambda(lrn)(l)
e = Conv2D(64,kernel_size=(7,7),activation='sigmoid',padding='same',data_format='channels_last',kernel_initializer=intial)(d)
f = Lambda(lrn)(e)
g = MaxPooling2D(pool_size=(3, 3),strides=2)(f)
ll = Activation('linear')(g)
#p = Lambda(pad)(g)
h = LocallyConnected2D(64,(5,5),activation='sigmoid',padding='valid',data_format='channels_last',kernel_initializer=intial)(ll)
#pp = Lambda(pad)(h)
i = LocallyConnected2D(32,(5,5),activation='sigmoid',padding='valid',data_format='channels_last',kernel_initializer=intial)(h)
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
        preprocessing_function=None,data_format=None,validation_split=0.1)

#x_train_1_batches = datagen.flow(x_train_1,y_train_1,batch_size=BATCH_SIZE_1)
datagen.fit(x_train_1)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               verbose=1,
                               min_lr=min_rate)

callbacks = [lr_reducer,keras.callbacks.TensorBoard(log_dir='./logs1',write_grads=True)] 

x_val_1_batches = datagen.flow_from_directory(directory='./ICDAR_reformat/1/train/',
										target_size=(48,48),
										color_mode='rgb',
										class_mode='categorical',
										shuffle=False,
    									subset='validation')

model1.fit_generator(datagen.flow_from_directory(directory='./ICDAR_reformat/1/train/',
					target_size=(48,48),
					color_mode='rgb',
					batch_size=32,
					class_mode='categorical',
					shuffle=True, subset='training'),
			          epochs=EPOCHS,
			          callbacks=callbacks,
			          steps_per_epoch=len(x_train_1)//BATCH_SIZE_1,
			          verbose=1,
			          validation_data=x_val_1_batches)



x_1_test = datagen.flow_from_directory(directory='./ICDAR_reformat/1/test/',
									target_size=(48,48),
									color_mode='rgb',
									class_mode='categorical',
									shuffle=False)

score = model1.evaluate_generator(generator=x_1_test,steps=len(x_1_test))
print('Test loss:', score[0])
print('Test accuracy:', score[1])

