import seaborn as sb
import time, os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, ZeroPadding2D, Activation, Lambda, Subtract
from keras.layers.normalization import BatchNormalization 
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from sklearn.preprocessing import LabelEncoder


# def weight_initialization(shape, name= None):
# 	return np.random.normal(loc= 0.0, scale=1e-2, size = shape)

# weights = weight_initialization((1000,1))



# def bias_initialization(shape, name= None):
# 	return np.random.normal(loc=0.5,scale=1e-2, size = shape)

input_shape = (200,200)

def Siamese_net():
	left_input = Input(input_shape)
	right_input = Input(input_shape)

	model = Sequential()
	model.add(Conv2D(64,(10,10),activation='relu',input_shape=input_shape))
	model.add(MaxPooling2D())
	model.add(Conv2D(128,(7,7),activation='relu'))
	model.add(MaxPooling2D())
	model.add(Conv2D(256,(4,4),activation='relu'))
	model.add(Flatten())
	model.add(Dense(4096,activation='sigmoid'))


	encoded_left = model(left_input)
	encoded_right = model(right_input)

	subtracted = Subtract()([encoded_left, encoded_right])
	both = Lambda(lambda x: abs(x))(subtracted)
	prediction = Dense(1,activation='sigmoid')(both)
	Siamese_net = Model(inputs=[left_input,right_input],outputs = prediction)

		


	optimizer = Adam(lr = 0.00001)
	Siamese_net.compile(loss='binary_crossentropy', optimizer=optimizer)
	print(Siamese_net.count_params())
	return Siamese_net