import torch
import torch.nn as nn
import torchvision #for datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.nn import functional as f
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

#%%  device configuration
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeuralNet(nn.Module):
     def __init__(self):
          super().__init__()
          self.conv1= nn.Conv2d(in_channels=1,out_channels=6,
                                kernel_size=(5,5),padding=(2,2))
          self.pool1= nn.AvgPool2d(kernel_size=(2,2),stride=2)
          
          self.conv2= nn.Conv2d(in_channels=6,out_channels=16,
                                kernel_size=(5,5))
          self.pool2= nn.AvgPool2d(kernel_size=(2,2),stride=2)
           
          self.FC1= nn.Linear(in_features=16*5*5,out_features=120)
          self.FC2= nn.Linear(in_features=120,out_features=84)
          self.FC3= nn.Linear(in_features=84,out_features=10) 
          
          # self.tanh= nn.Tanh()
          # self.relu= nn.ReLU()
          self.sigmoid= nn.Sigmoid()
          
     def forward(self,x):
          out= self.conv1(x)# 1*28*28
          out= self.sigmoid(out)#6*28*28
          out= self.pool1(out)# 6*14*14
          
          out= self.conv2(out)# 6*14*14
          out= self.sigmoid(out)#16*10*10
          out= self.pool2(out)# 16*5*5
                    
          out= self.FC1(out.flatten(start_dim=1))
          out= self.sigmoid(out)
          
          out= self.FC2(out)
          out= self.sigmoid(out)
          
          out= self.FC3(out)
          return out

#%% define hyperparameters
epoch=5 # number of training iteration
batch_size= 32 # the number of batch size is determind by the user, it helps training be easy and calculation fast
lr=0.001 # the value of learning rate is determined by the user,
model= NeuralNet()
#%% import MNIST dataset
train_dataset=MNIST(root='./data',
                    train=True,
                    transform=transforms.ToTensor(),
                    download=True)

test_dataset=MNIST(root='./data',
                   train=False,
                   transform=transforms.ToTensor(),
                   download=True)

train_loader= DataLoader(dataset=train_dataset,
                         shuffle=True,
                         batch_size=batch_size)

test_loader= DataLoader(dataset=test_dataset,
                        shuffle=False,
                        batch_size=batch_size)
   
# optimizer= optim.SGD(params=model.parameters(),lr=lr)
# optimizer= optim.SGD(params=model.parameters(),lr=lr,momentum=0.2)
optimizer= optim.Adam(params=model.parameters(),
                      lr=lr,betas=(0.9,0.999),eps= 1e-8)
# optimizer= optim.RMSprop(params=model.parameters(),lr=lr)
criteria= nn.CrossEntropyLoss()
#%% train Neural Network
mse=[]
for iter in range(1,epoch):
     er=[]
     for i,(xbatch,ybatch) in enumerate(train_loader):
          xbatch= xbatch.to(device)
          ybatch= ybatch.to(device)
          # farward pass
          ypred= model(xbatch) 
          # backward pass 
          loss=  criteria(ypred,ybatch)
          loss.backward() # triger gradient calculation
          optimizer.step()# update parameters(synaptic wieghts)
          optimizer.zero_grad() # clear gradients

          er.append(loss.detach())
          # print(i,er[i])
          
     loss= torch.mean(torch.tensor(er))
     mse.append(loss)
     print(f'MSE({iter-1}): {mse[iter-1]:.5f}')

mse=torch.tensor(mse)
#%% test trained Neural Network
with torch.no_grad():
    n_sample=0
    n_correct=0
    for i,(samples,labels) in enumerate(test_loader):
          samples= samples.to(device)
          labels= labels.to(device)
          # forward pass
          y_hat= model(samples)
          y_hat= torch.argmax(y_hat,dim=1)
          n_correct+=  torch.sum(y_hat==labels)
          n_sample+=labels.size(0)

accuracy= n_correct/n_sample *100
print('Accuracy:',accuracy)
