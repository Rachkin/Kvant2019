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
w = 48
h = 48

x = pd.read_csv('fer2013.csv')
data = x.values
y = data[:, 0]
pixels = data[:, 1]
X = np.zeros((pixels.shape[0], w*h))

for ix in range(X.shape[0]):
    p = pixels[ix].split(' ')
    for iy in range(X.shape[1]):
        X[ix, iy] = int(p[iy])
x = X.astype("float32")
x = x / 255
y = y.astype("long")
X_train = x[0:28710, :]
y_train = y[0:28710]
X_test = x[28710:30300, :]
y_test = y[28710:30300]

X_valid = x[30300:32300, :]
y_valid = y[30300:32300]


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

X_valid = torch.from_numpy(X_valid)
y_valid = torch.from_numpy(y_valid)

print(y_train[0])
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train.long())
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test.long())
X_valid = torch.FloatTensor(X_valid)
y_valid = torch.LongTensor(y_valid.long())

X_train = torch.reshape(X_train, (X_train.shape[0], 1 , w, h))
X_test = torch.reshape(X_test, (X_test.shape[0], 1 , w, h))
X_valid = torch.reshape(X_valid, (X_valid.shape[0], 1 , w, h))

import numpy as np
plt.imshow(X_valid[random.randint(0, 1000),0,:,:])
plt.show()