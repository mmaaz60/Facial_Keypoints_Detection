## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # Input image size (224, 224), gray scale image
        self.conv1 = nn.Conv2d(1, 32, 4)  # output size (32, 221, 221)
        
        # Pooling layer with kernel_size = 2, and stride = 2, output size (32, 110, 110)
        self.conv2 = nn.Conv2d(32, 64, 3)  # output size (64, 108, 108)
        
        # Pooling layer with kernel_size = 2, and stride = 2, output shape (64, 54, 54)
        self.conv3 = nn.Conv2d(64, 128, 2)  # output size (128, 53, 53)
        
        # Pooling layer with kernel_size = 2, and stride = 2, output shape (128, 26, 26)
        self.conv4 = nn.Conv2d(128, 256, 1)  # output size (256, 26, 26)
        
        # Pooling layer with kernel_size = 2, and stride = 2, output shape (256, 13, 13) 
        self.conv5 = nn.Conv2d(256, 512, 1)  # output size (512, 13, 13)
        
        # Pooling layer with kernel_size = 2, and stride = 2, output shape (512, 6, 6)
        self.fc1 = nn.Linear(512*6*6, 1024)  # output size (1024)
        self.fc2 = nn.Linear(1024, 512)  # output size (512)        
        self.fc3 = nn.Linear(512, 136)  # output size (136) 136/2 = 68 facial key points
        
        # pooling layer(s)
        self.pool = nn.MaxPool2d(2, 2)  # Pooling layer with kernel_size = 2, and stride = 2
        
        # dropout layer(s)
        self.drop1 = nn.Dropout(p=0.1)  # Dropout layer to avoid overfitting (dropout probability = 0.1)
        self.drop2 = nn.Dropout(p=0.2)  # Dropout layer to avoid overfitting (dropout probability = 0.2)
        self.drop3 = nn.Dropout(p=0.3)  # Dropout layer to avoid overfitting (dropout probability = 0.3)
        self.drop4 = nn.Dropout(p=0.4)  # Dropout layer to avoid overfitting (dropout probability = 0.4)
        self.drop5 = nn.Dropout(p=0.5)  # Dropout layer to avoid overfitting (dropout probability = 0.5)
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))
#         x = self.drop1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop1(x)
        x = self.pool(F.relu(self.conv3(x)))
#         x = self.drop3(x)
        x = self.pool(F.relu(self.conv4(x)))
#         x = self.drop4(x)
        x = self.pool(F.relu(self.conv5(x)))
        x = self.drop2(x)
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        # x = self.drop5(x)
        x = F.relu(self.fc2(x))
#         x = self.drop5(x)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
