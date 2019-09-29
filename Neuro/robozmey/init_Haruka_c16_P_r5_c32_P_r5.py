import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class resblock():
    def __init__(self):
        self.conv_1_1 = 1
        self.conv_1_2 = 1
        self.act_1 = 1
        self.batch_norm_1 = 1
        
        self.conv_2_1 = 1
        self.conv_2_2 = 1
        self.act_2 = 1
        self.batch_norm_2 = 1

class HarukaNet(torch.nn.Module):
    def __init__(self):
        super(HarukaNet, self).__init__()
        
        self.n_chanells = [0] * 3
        
        self.n_chanells[2] = 1
        
        self.n_chanells[0] = 16
        
        self.n_chanells[1] = 32
        
        self.n_resblock = 2
        
        self.n_resgroup = 2
        
        self.batch_norm0 = torch.nn.BatchNorm2d(1)

##############################################################################################
#       RES - BLOCKS
##############################################################################################
        self.conv_1 = [0] * self.n_resgroup
        self.conv_2 = [0] * self.n_resgroup
        self.act = [0] * self.n_resgroup
        self.batch_norm = [0] * self.n_resgroup
        self.pool = [0] * self.n_resgroup
        self.res = [0] * self.n_resgroup
        for i in range(self.n_resgroup):
            self.conv_1[i] = torch.nn.Conv2d(self.n_chanells[self.n_resgroup if (i-1)==-1 else i-1], self.n_chanells[i], 3, padding=1)
            self.conv_2[i] = torch.nn.Conv2d(self.n_chanells[i], self.n_chanells[i], 3, padding=1)
            self.act[i]  = torch.nn.ReLU()
            self.batch_norm[i] = torch.nn.BatchNorm2d(self.n_chanells[i])
            self.pool[i] = torch.nn.MaxPool2d(2, 2)
            self.res[i] = [resblock()] * self.n_resblock
            for j in range(self.n_resblock):
                self.res[i][j].conv_1_1 = torch.nn.Conv2d(self.n_chanells[i], self.n_chanells[i], 3, padding=1)
                self.res[i][j].conv_1_2 = torch.nn.Conv2d(self.n_chanells[i], self.n_chanells[i], 3, padding=1)
                self.res[i][j].act_1  = torch.nn.ReLU()
                self.res[i][j].batch_norm_1 = torch.nn.BatchNorm2d(self.n_chanells[i])
                
                self.res[i][j].conv_2_1 = torch.nn.Conv2d(self.n_chanells[i], self.n_chanells[i], 3, padding=1)
                self.res[i][j].conv_2_2 = torch.nn.Conv2d(self.n_chanells[i], self.n_chanells[i], 3, padding=1)
                self.res[i][j].act_2 = torch.nn.ReLU()
                self.res[i][j].batch_norm_2 = torch.nn.BatchNorm2d(self.n_chanells[i])
        

##############################################################################################
#       FULL C
##############################################################################################
        self.fc1   = torch.nn.Linear(12 * 12 * self.n_chanells[self.n_resgroup-1], 256)
        self.actf1  = torch.nn.Tanh()
        self.batch_normf1 = torch.nn.BatchNorm1d(256)
        
        self.fc2   = torch.nn.Linear(256, 64)
        self.actf2  = torch.nn.Tanh()
        self.batch_normf2 = torch.nn.BatchNorm1d(64)
        
        self.fc3   = torch.nn.Linear(64, 64)
        self.act_fc3  = torch.nn.Tanh()
        self.batch_norm_fc3 = torch.nn.BatchNorm1d(64)
        
        self.fc4   = torch.nn.Linear(64, 7)
        
        self.act_fc4  = torch.nn.Tanh()
    
    def forward(self, x):
        
        x = self.batch_norm0(x)
     
        for i in range( self.n_resgroup):
           # print(i)
            x = self.conv_1[i](x)
            x = self.conv_2[i](x)
            x = self.act[i](x)
            #x += t
            x = self.batch_norm[i](x)
            x = self.pool[i](x)
    
            for j in range(self.n_resblock):
               # print(str(i) + " " + str(j))
               # self.draw(x)
               # t = x
                x = self.res[i][j].conv_1_1(x)
                x = self.res[i][j].conv_1_2(x)
                x = self.res[i][j].act_1(x)
                x = self.res[i][j].batch_norm_1(x)
                
                x = self.res[i][j].conv_2_1(x)
                x = self.res[i][j].conv_2_2(x)
                x = self.res[i][j].act_2(x)
                x = self.res[i][j].batch_norm_2(x)
                
              #  x += t
        
#############################################
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
        x = self.act_fc4(x)
        
        return x
    
    def draw(self, x):
        import matplotlib.pyplot as plt
        plt.imshow(x.detach().numpy()[0,0,:,:])
        plt.show()
        
    def predict(self, x):
        #net.eval()
        with torch.no_grad():
            return self.forward(torch.reshape(x, (1, x.shape[0], x.shape[1], x.shape[2])))