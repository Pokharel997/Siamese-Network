import sys
import numpy as np
from PIL import Image
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import cv2


data_path = 'data'
train_path = 'data/train'
valid_path = 'data/val'

save_path= 'face_detect/'


# le = LabelEncoder()




def load_image(path):
	current_y = 0
	X =[]
	Y =[]

	for img_folder in os.listdir(path):
		image_folder_path = os.path.join(path,img_folder)
		for img in os.listdir(image_folder_path):
			image_path = os.path.join(image_folder_path,img)
			image = cv2.imread(image_path)
			image = cv2.resize(image,(200,200))
			X.append(np.stack(image))
			Y.append(current_y)
		current_y+=1
	X = np.stack(X)
	Y = np.vstack(Y)
	# Y = le.fit_transform(classes)
	# Y = to_categorical(Y,5)
	return X,Y



X,y  = load_image(valid_path)
print(X.shape)



# with open('test.pickle','wb') as f:
# 	pickle.dump((X,y),f)

	





