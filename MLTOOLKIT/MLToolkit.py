from torchvision import transforms
from enum import Enum
import torch.nn as nn
import torch
import torch.optim as optim
import sys
import tsk.MLTOOLKIT.NN_loader as nnl

def transform_image_to_tensor(images):
    """
    Load an Image seq from the specified path.
    
    Args:
    List[PIL images]: PAth to the directory containing image sequence.
    
    Returns:
    List[torch.Tensor] : list of image tensors
    """    
    transform = transforms.Compose([transforms.ToTensor()]) #Convert Image to tensor
    image_tensors = []
    for image in images:
        image_tensor = transform(image)
        image_tensors.append(image_tensor) 
    
    stacked_tensor = torch.cat(image_tensors, dim = 0)  #concatenate to shape [400,80,80]
    stacked_tensor = stacked_tensor.unsqueeze_(1)    #make dimension at index 1
    return stacked_tensor

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

class NN_Type(Enum):
    ConvNN = 1
    LinReg = 2

#Define the neural network 3 hidden layer regression model
class H_NeuralNet(object):
    """A utility class to hold my neural network from houdini.
    Calls various neural network models
    """

    def __init__(self, NN = NN_Type.ConvNN, list_dim = None, learning_rate = .001, n_epochs = 100) -> None:
        """initialize neural net based on user input
        Args:
            NN (enum, ): can be NN_type.ConvNN or NNType.LinReg. Defaults to NN_Type.ConvNN.
            list_dim (list, optional): dimensions for NN of type [in_1, out_1, out_2 ...]
            currently for CNN dim must be [in_c, out_c1, out_c2, output], for LinReg must be [in, out_1, out_2, out_3, output]
        """
        if NN == NN_Type.ConvNN :
            self.model = self.makeCNN(*list_dim)
        if NN == NN_Type.LinReg :
            self.model = self.makeLinearRegNN(*list_dim)
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.accuracy_list = []
        self.yhat_list = []

    def makeCNN(self, in_1, out_1, out_2, output):
        """make a convolutional neural network based on these parms

        Args:
            in_1 (_type_): no of input channels
            out_1 (_type_): no of output channels in first Conv
            out_2 (_type_): No of output channels in second conv
        """
        k = 5   #kernel
        p = 2   #padding
        s = 1   #stride
        nn1 = nnl.CNN(in_1, out_1, out_2, output, k, p, s)
        return nn1


    def makeLinearRegNN(self, in_1, out_1, out_2, out_3, output):
        """generate a neural network for linear regression with the input parms
        """
        nn1 = nnl.LinearRegressionModel(in_1, out_1, out_2, out_3, output)
        return nn1
    
    def update_nn(self, train_dataset, valid_dataset):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
        dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
        dataloader_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False)

        self.accuracy_list, self.yhat_list = train_model(self.n_epochs, optimizer, criterion, self.model, dataloader_train,
        dataloader_valid, self.accuracy_list, self.yhat_list)

        print(f"Accuracy is: {self.accuracy_list}")
        for item in self.yhat_list:
            print(f"predict = {item[0]} - value = {item[1]}.")

            
def conv_shape(x, k=1, p=0, s=1, d=1):
    """returns the shape of the fully connected layer 1

    Args:
        x (_type_): _description_
        k (int, optional): _description_. Defaults to 1.
        p (int, optional): _description_. Defaults to 0.
        s (int, optional): _description_. Defaults to 1.
        d (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
                                                                                  
def train_model(n_epochs, optimizer, criterion, model, dataloader_train,
                dataloader_valid, accuracy_list, yhat_list):
    cost_list = []
    N_test = len(dataloader_valid)
    print(f"\n\nSTARTING TRAINING with \n n_epochs = {n_epochs} \
          \n Training Items = {len(dataloader_train)}\n Valid Items = {N_test}.")
    COST = 0
    for epoch in range(n_epochs):
        COST=0
        for i, (x, y) in enumerate(dataloader_train):
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            COST+=loss.data
            #print loss every 25 epochs
            if (epoch + 1) % 25 == 0:
                print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")
        
        cost_list.append(COST)
        correct=0
        #perform a prediction on the validation  data  
        for x_test, y_test in dataloader_valid:
            z = model(x_test)
            yhat_list.append((z.data.numpy()[0], y_test.numpy()[0]))
            correct += (abs(z.data-y_test) < .05).sum().item()
        accuracy = correct / N_test
        accuracy_list.append(accuracy)
        
    return accuracy_list, yhat_list
    
        
                       