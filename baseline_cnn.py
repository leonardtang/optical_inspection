import pandas as pd
import numpy as np

from skimage.io import imread
import matplotlib.pyplot as plt
# %matplotlib inline

import torch
import torchvision
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d
from torch.optim import Adam, SGD


def load_dataset():
    """ Loads data, classes broken down by folder """
    data_path = "//NEU_surface_defect image"
    dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor()
    )
    return dataset  # Returns iterable DataSet object


# Split data as 70/15/15
full_dataset = load_dataset()
train_size = int(0.7 * len(full_dataset))
val_size = int((len(full_dataset) - train_size) / 2)
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_set, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])

# Converting DataSet to DataLoader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=4, shuffle=True)

# # Visualizing data -- [3, 200, 200]
# for i in range(len(train_dataset)):
#     image = train_dataset.__getitem__(i)
#     print(i, image[0].size(), "", "Target:", image[1])


class Net(Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining one 2D convolution layer
            # Conv output is
            Conv2d(in_channels=3, out_channels=40, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(40),
            ReLU(inplace=True),
            # MaxPool output is
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            # Conv output is 94 x 94 x 32
            Conv2d(in_channels=40, out_channels=20, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(20),
            ReLU(inplace=True),
            # MaxPool output is 47 x 47 x 32
            MaxPool2d(kernel_size=2, stride=2),
        )

        # Vanilla neural network perceptron layer
        self.linear_layers = Sequential(
            # (in_features, out_features = predicted labels, bias)
            Linear(20 * 12 * 12, 6),
            ReLU(inplace=True)
        )

        self.output_layer = Sequential(
            # (dimension to compute softmax along)
            # Output shape equals input shape
            Softmax(dim=1)
        )

    # Feeding training input x forward in NN
    def forward(self, x):
        # print(x.size())
        x = self.cnn_layers(x)
        # print(x.size())
        # Have to flatten before linear layer
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        # print(x.size())
        x = self.output_layer(x)
        return x

# print(model)
# print(train_dataset.__getitem__(0).__len__(), "\n", train_dataset[1].__len__())
# print(train_dataset[0])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net()
optimizer = Adam(model.parameters(), lr=0.01)
criterion = CrossEntropyLoss()
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

# Training the network
n_epochs = 10
train_losses, val_losses = [], []
for epoch in range(n_epochs):
    print("Epoch:", epoch + 1)
    # Iterating through training set
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        print("Iteration:", i + 1)
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        inputs, labels = inputs.to(device), labels.to(device)
        # Set gradients of all layers' parameters to zero (not done automatically)
        optimizer.zero_grad()
        # Feeding forward
        outputs = model(inputs)
        # Backpropogation
        loss = criterion(outputs, labels)
        loss.backward()
        # Adam optimizer -- in general, SGD or similar
        optimizer.step()
        running_loss += loss.item()
        train_losses.append(loss)
        if (i + 1) % 10 == 0:
            print("Train loss:", loss.item())

    # At the end of the epoch, do a pass on the validation set
    running_val_loss = 0
    accuracy = 0
    for inputs, labels in val_loader:
        inputs, labels = Variable(inputs), Variable(labels)
        inputs, labels = inputs.to(device), labels.to(device)

        val_outputs = model(inputs)
        val_loss = criterion(val_outputs, labels)
        running_val_loss += val_loss.item()
        val_losses.append(val_loss)

    # Printing epoch summary
    print("Validation loss = {:.2f}".format(running_val_loss / len(val_loader)))


plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.show()




