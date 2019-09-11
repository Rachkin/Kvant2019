import torch
import random
import numpy as np
import matplotlib.pyplot as plt

class HarukaNet(torch.nn.Module):
    def __init__(self):
        super(HarukaNet, self).__init__()
        
        self.n_chanells = 16
        
        self.n_ar_layers = 5
        
        self.batch_norm0 = torch.nn.BatchNorm2d(1)

        self.conv1_1 = torch.nn.Conv2d(1, self.n_chanells, 3, padding=1)
#        self.conv1_2 = torch.nn.Conv2d(8, 8, 3, padding=1)
        self.act1  = torch.nn.ReLU()
        self.batch_norm1 = torch.nn.BatchNorm2d(self.n_chanells)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        
        self.conv2_1 = torch.nn.Conv2d(self.n_chanells, self.n_chanells, 3, padding=1)
#        self.conv2_2 = torch.nn.Conv2d(8, 8, 3, padding=1)
        self.act2  = torch.nn.ReLU()
        self.batch_norm2 = torch.nn.BatchNorm2d(self.n_chanells)
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        
        self.conv_ar_1 = [0 for c in range(self.n_ar_layers)]
        self.act_ar_1 = [0 for c in range(self.n_ar_layers)]
        self.batch_norm_ar_1 = [0 for c in range(self.n_ar_layers)]
        
        self.conv_ar_2 = [0 for c in range(self.n_ar_layers)]
        self.act_ar_2 = [0 for c in range(self.n_ar_layers)]
        self.batch_norm_ar_2 = [0 for c in range(self.n_ar_layers)]
        
        for i in range(self.n_ar_layers):
            self.conv_ar_1[i] = torch.nn.Conv2d(self.n_chanells, self.n_chanells, 3, padding=1)
#           self.conv3_2 = torch.nn.Conv2d(16, 16, 3, padding=1)
            self.act_ar_1[i]  = torch.nn.ReLU()
            self.batch_norm_ar_1[i] = torch.nn.BatchNorm2d(self.n_chanells)
            
            self.conv_ar_2[i] = torch.nn.Conv2d(self.n_chanells, self.n_chanells, 3, padding=1)
#           self.conv3_2 = torch.nn.Conv2d(16, 16, 3, padding=1)
            self.act_ar_2[i]  = torch.nn.ReLU()
            self.batch_norm_ar_2[i] = torch.nn.BatchNorm2d(self.n_chanells)
            
#
        self.fc1   = torch.nn.Linear(12 * 12 * self.n_chanells, 256)
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
        for i in range(self.n_ar_layers):
            
            t = x
            x = self.conv_ar_1[i](x)
       
            x = self.act_ar_1[i](x)
            x = self.batch_norm_ar_1[i](x)
            
            x = self.conv_ar_2[i](x)

            x = self.act_ar_2[i](x)
            x = self.batch_norm_ar_2[i](x)
            
            x += t
        
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
    def predict(self, x):
        #net.eval()
        with torch.no_grad():
            return self.forward(torch.reshape(x, (1, x.shape[0], x.shape[1], x.shape[2])))