import pickle

X = pickle.load(open("X3.pickle", "rb"))
Y = pickle.load(open("Y3.pickle", "rb"))

from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input,decode_predictions
from keras import backend as K
from keras.layers import add, Conv2D,MaxPooling2D,UpSampling2D,Input,BatchNormalization, RepeatVector, Reshape
from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator

def NoiseModel(input):

  model = Conv2D(16,(3,3), activation='relu',padding='same',strides=1)(input)
  model = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(model)
  model = Conv2D(64,(2,2), activation='relu',padding='same',strides=1)(model)
  model = Conv2D(128,(2,2), activation='relu',padding='same',strides=1)(model)

  model = Conv2D(128,(2,2), activation='relu',padding='same',strides=1)(model)
  model = Conv2D(64,(2,2), activation='relu',padding='same',strides=1)(model)
  model = Conv2D(32,(2,2), activation='relu',padding='same',strides=1)(model)
  model = Conv2D(16,(2,2), activation='relu',padding='same',strides=1)(model)
  model = Conv2D(3,(2,2), activation='relu',padding='same',strides=1)(model)

  return model

Input1 = Input(shape=(300, 300, 3))
Output = NoiseModel(Input1)
Model = Model(inputs=Input1, outputs=Output)

Model.compile(optimizer='adam', loss='mean_squared_error')

def GenerateInputs(X,y):
    for i in range(len(X)):
        X_input = X[i].reshape(1,300,300,3)
        y_input = y[i].reshape(1,300,300,3)
        yield (X_input,y_input)

final = Model.fit_generator(GenerateInputs(X,Y), epochs=50, steps_per_epoch=68, verbose=1)
