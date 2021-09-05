import opendatasets as od
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.util import random_noise
import os # miscellaneous operation system interfaces
import pathlib
from os import listdir
from os.path import isfile, join

#dataset
od.download("https://www.kaggle.com/duttadebadri/image-classification")

#Example of how to add noise to image
#   img = cv2.imread("/content/image-classification/images/images/travel and  adventure/Places365_val_00005731.jpg")
#    cv2.imshow(img) #before
#    noise_img = random_noise(img, mode='gaussian',seed=None)
#    noise_img = np.array(255*noise_img, dtype = 'uint8')
#    cv2.imshow(noise_img) #after

X = []
Y = []
for file in os.listdir("/content/image-classification/images/images/travel and  adventure"):
  img = cv2.imread("/content/image-classification/images/images/travel and  adventure/" + file)
  img = cv2.resize(img, (300, 300))
  noise_img = random_noise(img, mode='gaussian',seed=None)
  noise_img = np.array(255*noise_img, dtype = 'uint8')
  X.append(noise_img)
  Y.append(img)
  
X = np.array(X)
Y = np.array(Y)

import pickle

pickle_out = open("X3.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("Y3.pickle", "wb")
pickle.dump(Y, pickle_out)
pickle_out.close()
