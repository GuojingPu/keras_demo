#!/usr/bin/python3
# -*- coding:utf-8 -*-
from keras.datasets import cifar10
import  numpy as np
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D


# import os
#
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#
# os.environ["CUDA_VISIBLE_DEVICES"]="0"


"""保存在   /Users/pgj/.keras/datasets/cifar-10-batches-py/ """
(x_img_train,y_lable_train),(x_img_test,y_lable_test) = cifar10.load_data()
print("x_img_train:",x_img_train.shape)
print("y_lable:",y_lable_train.shape)
print("x_img_test:",x_img_test.shape)
print("y_lable_test:",y_lable_test.shape)

x_img_train_normalize = x_img_train.astype('float32')/255.0
x_img_test_normalize = x_img_test.astype('float32')/255.0

y_lable_one_hot = np_utils.to_categorical(y_lable_train)
y_lable_test_one_hot = np_utils.to_categorical(y_lable_test)

print("y_lable_one_hot:",y_lable_one_hot.shape)
print("y_lable_test_one_hot",y_lable_test_one_hot.shape)


model = Sequential()

model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(32,32,3),activation='relu',padding='same'))
model.add(Dropout(rate=0.25))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
model.add(Dropout(rate=0.25))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(rate=0.25))

model.add(Dense(1024,activation='relu'))
model.add(Dropout(rate=0.25))

model.add(Dense(10,activation='relu'))


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])



train = model.fit(x=x_img_train_normalize,y=y_lable_one_hot,validation_split=0.2,epochs=10,verbose=2)



print(model.summary())