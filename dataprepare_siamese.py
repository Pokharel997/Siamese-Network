import sys
import numpy as np
from PIL import Image
import pickle
import os
import matplotlib.pyplot as plt


data_path = 'actual_data'
train_path = 'actual_data/train'
valid_path = 'actual_data/valid'

save_path= 'actual_data/'



def load_image(path):
	X = []
	Y = []

	for img_folder in os.listdir(path):
		image_folder_path = os.path.join(path,img_folder)
		for img in os.listdir(image_folder_path):
			image_path = os.path.join(image_folder_path,img)
			image = Image.open(image_path)
			pixels = np.asarray(image)
			X.append(pixels)
	return X	

img_pixels = load_image(train_path)
print(img_pixels)




