#Define the neural network 3 hidden layer regression model
import torch.nn as nn
import torch
class LinearRegressionModel(nn.Module):
    def __init__(self, in_1 = 1, out_1 = 64, out_2 = 128, out_3 = 64, output = 1):
        super(LinearRegressionModel, self).__init__()
        self.hidden1 = nn.Linear(in_1,out_1)
        self.hidden2 = nn.Linear(out_1, out_2)
        self.hidden3 = nn.Linear(out_2, out_3)
        self.output = nn.Linear(out_3, output)
        
    def forward(self,x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = self.output(x)
        return x


class CNN(nn.Module):
    def __init__(self, in_1 = 1, out_1 = 8, out_2=16, output = 1, k = 5, p = 2, s = 1):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=in_1, out_channels=out_1, kernel_size=k, padding=p) #define Neural network layer
        self.bn1 = nn.BatchNorm2d(out_1) #num of channels
        self.maxpool1=nn.MaxPool2d(kernel_size=2)    
        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=k, stride=s, padding=p)
        self.bn2 = nn.BatchNorm2d(out_2) #num out channels
        self.maxpool2=nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(out_2 * 20 * 20, 128)
        self.fc2 = nn.Linear(128, output)

            # Prediction
    def forward(self, x):
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.bn1(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.bn2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
        
    # Outputs in each steps
    def activations(self, x):
        #outputs activation this is not necessary
        z1 = self.cnn1(x)
        a1 = torch.relu(z1)
        out = self.maxpool1(a1)
        
        z2 = self.cnn2(out)
        a2 = torch.relu(z2)
        out1 = self.maxpool2(a2)
        out = out.view(out.size(0),-1)
        return z1, a1, z2, a2, out1,out

