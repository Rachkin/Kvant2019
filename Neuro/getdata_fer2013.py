import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Conv2D
from keras import backend as K
from keras.layers import Dense, MaxPooling2D,  Flatten, Dropout
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

import torchvision.datasets

#torch.pytorch.set_default_tensor_type('torch.DoubleTensor')


x = pd.read_csv('fer2013.csv')
data = x.values
y = data[:, 0]
pixels = data[:, 1]
X = np.zeros((pixels.shape[0], 48*48))

for ix in range(X.shape[0]):
    p = pixels[ix].split(' ')
    for iy in range(X.shape[1]):
        X[ix, iy] = int(p[iy])
x = X.astype("float32")
x = x / 255
y = y.astype("long")
X_train = x[0:28710, :]
y_train = y[0:28710]
X_test = x[28710:32300, :]
y_test = y[28710:32300]


datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range = 0.0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False)

#datagen.fit(X_train)

X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)

X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)

print(y_train[0])
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train.long())
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test.long())

X_train = torch.reshape(X_train, (X_train.shape[0], 1 , 48, 48 ))
X_test = torch.reshape(X_test, (X_test.shape[0], 1 ,48, 48))