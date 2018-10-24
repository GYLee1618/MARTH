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
from keras.models import Model
from keras.layers import *
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import categorical_accuracy
import random
import time
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


BATCH_SIZE = 32
NUM_CLASSES = 31
EPOCHS = 1000

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

inp = Input(shape=input_shape)
conv1 = Conv2D(32, kernel_size=(2, 2),activation='relu',padding='same')(inp)
bn1 = BatchNormalization()(conv1)
rl1 = Activation('relu')(bn1)
do1 = Dropout(0.3)(rl1)
conv2 = Conv2D(32,kernel_size=(4,4),activation='relu',padding='same')(do1)
bn2 = BatchNormalization()(conv2)
rl2 = Activation('relu')(bn2)
mp1 = MaxPooling2D(pool_size=(2, 2),strides=2)(rl2)

#model.add(Conv2D(48, kernel_size=(3, 3),activation='relu',padding='same'))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(Conv2D(48, kernel_size=(5, 5),activation='relu',padding='same'))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(3, 3),strides=2))

conv3 = Conv2D(64, kernel_size=(4, 4),activation='relu',padding='same')(mp1)
bn3 = BatchNormalization()(conv3)
rl3 = Activation('relu')(bn3)
do2 = Dropout(.4)(rl3)
conv4 = Conv2D(64, kernel_size=(2, 2),activation='relu',padding='same')(do2)
bn4 = BatchNormalization()(conv4)
rl4 = Activation('relu')(bn4)
mp2 = MaxPooling2D(pool_size=(2, 2),strides=2)(rl4)

conv5 = Conv2D(128, kernel_size=(2, 2),activation='relu',padding='same')(mp2)
bn5 = BatchNormalization()(conv5)
rl5 = Activation('relu')(bn5)
do3 = Dropout(.25)(rl5)
conv6 = Conv2D(128, kernel_size=(3, 3),activation='relu',padding='same')(do3)
bn6 = BatchNormalization()(conv6)
rl6 = Activation('relu')(bn6)
mp6 = MaxPooling2D(pool_size=(2, 2),strides=2)(rl6)
fl = Flatten()(mp6)

# shl3 = fl(mp6(rl6(bn6(conv6(do3(rl5(bn5(conv5))))))))

#model.add(Dense(48, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01)))
dense1 = Dense(NUM_CLASSES, activation='softmax')(fl)
dense2 = Dense(NUM_CLASSES, activation='softmax')(fl)

model1 = Model(inp,dense1)
model2 = Model(inp,dense2)

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

losses1 = []
losses2 = []
for ii in range(EPOCHS):
  learn = lr_schedule(ii)
  optim = keras.optimizers.SGD(lr=learn)

  model1.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=optim,
          metrics=['accuracy'])

  model2.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=optim,
          metrics=['accuracy'])

  cooldown -= 1
  print("Epoch {}/{}".format(ii+1,EPOCHS))
  x_train_1_batches = datagen.flow(x_train_1,y_train_1,batch_size=32)
  x_train_2_batches = datagen.flow(x_train_2,y_train_2,batch_size=32)
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
      history = model1.fit(x_train_1_b, y_train_1_b,epochs=1,verbose=0)
      train1error_sum += history.history['loss'][0]
      train1acc_sum += history.history['acc'][0]
      batch1_count +=1

    elif (rng > .5 and batch2_count < len(x_train_2_batches)):
      x_train_2_b,y_train_2_b = x_train_2_batches[batch2_count]
      history = model2.fit(x_train_2_b,y_train_2_b,epochs=1,verbose=0)
      train2error_sum += history.history['loss'][0]
      train2acc_sum += history.history['acc'][0]
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

score = model1.test_on_batch(xtest0, ytest0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
score = model2.test_on_batch(xtest1, ytest1)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


