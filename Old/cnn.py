# cnn.py
import torch.nn as nn
import torch.nn.functional as F

# TODO: These layer sizes were pulled out of thin air because PRIONN doesn't provide that level of detail. Make sure the sizes are appropriate for the task at hand.

class CnnRegressor(nn.Module):
    """ Define the CNN model. """

    def __init__(self, batch_size, inputs, outputs):
        # initialization of the superclass
        super(CnnRegressor, self).__init__()

        # store the parameters
        self.batch_size = batch_size
        self.inputs = inputs
        self.outputs = outputs

        # define the input layer
        self.input_layer = nn.Conv1d(inputs, batch_size, 1, stride = 1)
    
        # define max pooling layer
        self.max_pooling_layer = nn.MaxPool1d(1)

        # define other convolutional layers
        self.conv_layer1 = nn.Conv1d(batch_size, 256, 1, stride = 3)
        self.conv_layer2 = nn.Conv1d(256, 512, 1, stride = 3)
        self.conv_layer3 = nn.Conv1d(512, 1024, 1, stride = 3)

        # define the flatten layer
        self.flatten_layer = nn.Flatten()

        # Define the fully connected layers
        self.full1 = nn.Linear(1024,  512)
        self.full2 = nn.Linear(512,  256)
        self.full3 = nn.Linear(256,  128)

        # define the output layer
        self.output_layer = nn.Linear(128, outputs)

    # define the method to feed the inputs to the model
    def forward(self, input):
        # input is reshaped to the 1D array and fed into the input layer
        input = input.reshape((self.batch_size, self.inputs, 1))

        # ReLU is applied on the output of input layer
        output = F.relu(self.input_layer(input))

        # max pooling is applied and then Convolutions are done with ReLU
        output = self.max_pooling_layer(output)
        output = F.relu(self.conv_layer1(output))

        output = self.max_pooling_layer(output)
        output = F.relu(self.conv_layer2(output))

        output = self.max_pooling_layer(output)
        output = F.relu(self.conv_layer3(output))

        # flatten layer is applied
        output = self.flatten_layer(output)

        # linear layer and ReLu is applied
        output = F.relu(self.full1(output))
        output = F.relu(self.full2(output))
        output = F.relu(self.full3(output))

        # finally, output layer is applied
        output = self.output_layer(output)
        return output
