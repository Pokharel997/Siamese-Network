import sys
import numpy as np
from PIL import Image
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


data_path = 'face_samples'
train_path = 'face_samples/train'
valid_path = 'face_samples/val'

save_path= 'face_detect/'


le = LabelEncoder()

X = []
y = []

def load_image(path):
	X = []
	classes = []

	for img_folder in os.listdir(path):
		image_folder_path = os.path.join(path,img_folder)
		for img in os.listdir(image_folder_path):
			image_path = os.path.join(image_folder_path,img)
			image = Image.open(image_path)
			pixels = np.asarray(image)
			classes.append(img_folder)
			X.append(np.stack(pixels))

	# Y = le.fit_transform(classes)
	# Y = to_categorical(Y,5)
	return X,classes

try:
    X,y  = load_image(valid_path)
except:
    raise

with open(os.path.join(save_path,"test2.pickle"),"wb") as f:
	pickle.dump((X,y),f)