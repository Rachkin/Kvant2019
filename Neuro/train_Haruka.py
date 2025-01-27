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
class HarukaNet(torch.nn.Module):
    def __init__(self):
        super(HarukaNet, self).__init__()
        
        n_chanells = 32
        
        self.batch_norm0 = torch.nn.BatchNorm2d(1)

        self.conv1_1 = torch.nn.Conv2d(1, n_chanells, 3, padding=1)
#        self.conv1_2 = torch.nn.Conv2d(8, 8, 3, padding=1)
        self.act1  = torch.nn.ReLU()
        self.batch_norm1 = torch.nn.BatchNorm2d(n_chanells)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        
        self.conv2_1 = torch.nn.Conv2d(n_chanells, n_chanells, 3, padding=1)
#        self.conv2_2 = torch.nn.Conv2d(8, 8, 3, padding=1)
        self.act2  = torch.nn.ReLU()
        self.batch_norm2 = torch.nn.BatchNorm2d(n_chanells)
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        
        self.conv3_1 = torch.nn.Conv2d(n_chanells, n_chanells, 3, padding=1)
#        self.conv3_2 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.act3  = torch.nn.ReLU()
        self.batch_norm3 = torch.nn.BatchNorm2d(n_chanells)
        
        self.conv4_1 = torch.nn.Conv2d(n_chanells, n_chanells, 3, padding=1)
#        self.conv4_2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.act4  = torch.nn.ReLU()
        self.batch_norm4 = torch.nn.BatchNorm2d(n_chanells)
#        
        self.conv5_1 = torch.nn.Conv2d(n_chanells, n_chanells, 3, padding=1)
#        self.conv5_2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.act5  = torch.nn.ReLU()
        self.batch_norm5 = torch.nn.BatchNorm2d(n_chanells)
        
        self.conv6_1 = torch.nn.Conv2d(n_chanells, n_chanells, 3, padding=1)
#        self.conv6_2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.act6  = torch.nn.ReLU()
        self.batch_norm6 = torch.nn.BatchNorm2d(n_chanells)
        
        self.conv7_1 = torch.nn.Conv2d(n_chanells, n_chanells, 3, padding=1)
#        self.conv6_2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.act7  = torch.nn.ReLU()
        self.batch_norm7 = torch.nn.BatchNorm2d(n_chanells)
        
        self.conv8_1 = torch.nn.Conv2d(n_chanells, n_chanells, 3, padding=1)
#        self.conv6_2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.act8  = torch.nn.ReLU()
        self.batch_norm8 = torch.nn.BatchNorm2d(n_chanells)
        
        self.conv9_1 = torch.nn.Conv2d(n_chanells, n_chanells, 3, padding=1)
#        self.conv6_2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.act9  = torch.nn.ReLU()
        self.batch_norm9 = torch.nn.BatchNorm2d(n_chanells)
        
        self.conv10_1 = torch.nn.Conv2d(n_chanells, n_chanells, 3, padding=1)
#        self.conv6_2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.act10  = torch.nn.ReLU()
        self.batch_norm10 = torch.nn.BatchNorm2d(n_chanells)
        
        self.conv11_1 = torch.nn.Conv2d(n_chanells, n_chanells, 3, padding=1)
#        self.conv6_2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.act11  = torch.nn.ReLU()
        self.batch_norm11 = torch.nn.BatchNorm2d(n_chanells)
#
        self.fc1   = torch.nn.Linear(12 * 12 * n_chanells, 256)
        self.actf1  = torch.nn.Tanh()
        self.batch_normf1 = torch.nn.BatchNorm1d(256)
        
        self.fc2   = torch.nn.Linear(256, 64)
        self.actf2  = torch.nn.Tanh()
        self.batch_normf2 = torch.nn.BatchNorm1d(64)
        
        self.fc3   = torch.nn.Linear(64, 64)
        self.act_fc3  = torch.nn.Tanh()
        self.batch_norm_fc3 = torch.nn.BatchNorm1d(64)
        
        self.fc4   = torch.nn.Linear(64, 7)
    
    def forward(self, x):
       # x = x.float(x)
        x = self.batch_norm0(x)
     #   t = x
        x = self.conv1_1(x)
       # x = self.conv1_2(x)
        x = self.act1(x)
        #x += t
        x = self.batch_norm1(x)
        x = self.pool1(x)
#############################################        
        t = x
        x = self.conv2_1(x)
#        x = self.conv2_2(x)
        x += t
        x = self.act2(x)
        x = self.batch_norm2(x)
        x = self.pool2(x)
#############################################        
        t = x
        x = self.conv3_1(x)
#        x = self.conv3_2(x)
        x += t
        x = self.act3(x)
        x = self.batch_norm3(x)
#############################################       
        t = x
        x = self.conv4_1(x)
#        x = self.conv4_2(x)
        x = self.act4(x)
#        x += t
        x = self.batch_norm4(x)
        
#        t = x
        x = self.conv5_1(x)
#        x = self.conv5_2(x)
        x += t
        x = self.act5(x)
        x = self.batch_norm5(x)
#############################################        
        t = x
        x = self.conv6_1(x)
#        x = self.conv6_2(x)
        x = self.act6(x)
#        x += t
        x = self.batch_norm6(x)
        
#        t = x
        x = self.conv7_1(x)
#        x = self.conv6_2(x)
        x += t
        x = self.act7(x) 
        x = self.batch_norm7(x)
#############################################       
        t = x
        x = self.conv8_1(x)
#        x = self.conv6_2(x)
        x = self.act8(x)
#        x += t
        x = self.batch_norm8(x)
        
#        t = x
        x = self.conv9_1(x)
#        x = self.conv6_2(x)
        x += t
        x = self.act9(x)
        x = self.batch_norm9(x)
#############################################       
        t = x
        x = self.conv10_1(x)
#        x = self.conv6_2(x)
        x = self.act10(x)
#        x += t
        x = self.batch_norm10(x)
        
#        t = x
        x = self.conv11_1(x)
#        x = self.conv6_2(x)
        x += t
        x = self.act11(x)
        x = self.batch_norm11(x)
#############################################        
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.fc1(x)
        x = self.actf1(x)
        x = self.batch_normf1(x)
        ####################################
        x = self.fc2(x)
        x = self.actf2(x)
        x = self.batch_normf2(x)
        ####################################
        x = self.fc3(x)
        x = self.act_fc3(x)
        x = self.batch_norm_fc3(x)
        ####################################
        x = self.fc4(x)
        
        return x
#############################################
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
#device = torch.device('cuda:0')

def train(net, X_train, y_train, X_test, y_test):
    net = net.to(device)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1.0e-3)
    
    batch_size = 100

    test_accuracy_history = []
    test_loss_history = []

    X_test = X_test.to(device)
    y_test = y_test.to(device)

    for epoch in range(10):
        order = np.random.permutation(len(X_train))
        for start_index in range(0, len(X_train), batch_size):
            optimizer.zero_grad()
            net.train()

            batch_indexes = order[start_index:start_index+batch_size]

            X_batch = X_train[batch_indexes].to(device)
            y_batch = y_train[batch_indexes].to(device)
            
            preds = net.forward(X_batch) 

            loss_value = loss(preds, y_batch)
            loss_value.backward()

            optimizer.step()

        net.eval()
        with torch.no_grad():
            test_preds = net.forward(X_test)
            test_loss_history.append(loss(test_preds, y_test).data.cpu())
    
            accuracy = (test_preds.argmax(dim=1) == y_test).float().mean().data.cpu()
            test_accuracy_history.append(accuracy)
    
            print(accuracy)
            print(test_loss_history[-1])
            print("==============")
    print('---------------')
    return test_accuracy_history, test_loss_history
##############################################
accuracies = {}
losses = {}

net = HarukaNet()

accuracies['='], losses['='] = \
    train(net, 
          X_train, y_train, X_test, y_test)

import matplotlib.pyplot as plt
for experiment_id in accuracies.keys():
    plt.plot(accuracies[experiment_id], label=experiment_id)
plt.legend()
plt.title('Validation Accuracy');
print("1")
import matplotlib.pyplot as plt
for experiment_id in losses.keys():
    plt.plot(losses[experiment_id], label=experiment_id)
plt.legend()
plt.title('Validation Loss');
########################################
X_valid = X_valid.to(device)
y_valid = y_valid.to(device)

net.eval()
with torch.no_grad():
    test_preds = net.forward(X_valid)
    accuracy = (test_preds.argmax(dim=1) == y_valid).float().mean().data.cpu()
    print(accuracy)