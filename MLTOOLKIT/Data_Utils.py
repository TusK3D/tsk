import numpy as np
import hou
import torch
import torch.nn as nn
import matplotlib.pyplot as plt




def convert_float_to_tensor(data):
    """Converts an array of floating pt values to a Pytorch Tensor
    Args:
        data (array): an array of floats 
    Returns:
        tensor of floats
    """
    return torch.tensor(data, dtype=torch.float32).reshape(len(data), -1)


def convert_vector_to_tensor(data):
    """Converts an array of vectors to a Tensor

    Args:
        data (array): an array of vectors


    """

def split_data_train(
        X,Y, train_prob = 0.7, valid_prob = 0.2, test_prob = 0.1
        ):
    generator1 = torch.Generator().manual_seed(126)
    my_dataset = torch.utils.data.TensorDataset(X,Y)
    # return DataLoader(my_dataset)
    train_set, valid_set, test_set = torch.utils.data.random_split(
        my_dataset, [train_prob, valid_prob, test_prob])
    return train_set, valid_set, test_set
    


#Define the neural network 3 hidden layer regression model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.hidden1 = nn.Linear(1,64)
        self.hidden2 = nn.Linear(64,128)
        self.hidden3 = nn.Linear(128,64)
        self.output = nn.Linear(64,1)
        
    def forward(self,x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = self.output(x)
        return x
    

    


        
        
