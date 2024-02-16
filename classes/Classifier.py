import torch
import torch.nn as nn
from torch.functional import F

# Define the CNN model
class CNN(nn.Module):
    def __init__(self, kernel_sizes, out_channels, dropout_prob, input_size):
        super(CNN, self).__init__()

        output_size = 2
        # Define the input channel
        input_channel = 1

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        output_dim = 0
        for i in range(len(kernel_sizes)):
            for j in range(len(out_channels)):
                output_dim += out_channels[j]
                padding = (kernel_sizes[i] - 1) // 2
                self.conv_layers.append(nn.Conv2d(
                    input_channel, out_channels[j], (kernel_sizes[i], input_size), padding=(padding, 0)))
                
        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Fully connected layer
        self.fc1 = nn.Linear(output_dim, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        # Convolutional layers
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv_layers]

        # Max pooling
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        # Concatenate the outputs of the convolutional layers
        x = torch.cat(x, 1)

        # Dropout
        x = self.dropout(x)

        # Fully connected layer
        x = self.fc1(x)

        return F.log_softmax(x, dim=1)
