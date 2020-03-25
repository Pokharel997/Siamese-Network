import sys
import numpy as np
from PIL import Image
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


data_path = 'data'
train_path = 'data/train'
valid_path = 'data/valid'

# save_path= 'face_detect/'


le = LabelEncoder()


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



X,y  = load_image(train_path)

print(X,y)

# with open(os.path.join(save_path,"train.pickle"),"wb") as f:
# 	pickle.dump((X,y),f)

	





