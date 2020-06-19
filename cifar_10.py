import time
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn


def pre_processing_and_samples():
    # For reproducible data results -- same set of numbers appear each time
    seed = 42
    np.random.seed(seed)  # For calling np.random()
    torch.manual_seed(seed)  # For calling torch.rand()

    # transforms.ToTensor() converts our PILImage to a tensor of shape (C x H x W) in the range [0,1]
    # transforms.Normalize(mean,std) normalizes a tensor to a (mean, std) for (R, G, B)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Downloading CIFAR10 dataset using Pytorch
    # The compose function allows for multiple transforms
    train_set = torchvision.datasets.CIFAR10(root='./cifardata', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./cifardata', train=False, download=True, transform=transform)

    # Image dimensions: (C x H x W) = (3 x 32 x 32)

    # CIFAR10 image categories
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Training sampler -- holds random indices later used to access training set
    number_training_samples = 2000
    train_sampler = SubsetRandomSampler(np.arange(number_training_samples, dtype=np.int64))

    # Validation sampler
    # The range is not (0, n) since the indices here are part of one training dataset
    number_val_samples = 5000
    val_sampler = SubsetRandomSampler(
        np.arange(number_val_samples, dtype=np.int64))

    # Test sampler
    number_test_samples = 5000
    test_sampler = SubsetRandomSampler(np.arange(number_val_samples, number_val_samples + number_test_samples, dtype=np.int64))

    return [train_set, test_set, train_sampler, val_sampler, test_sampler]


def outputSize(in_size, kernel_size, stride, padding):
    """ For calculating conv2d or pooling layer output dimensions """
    output = int((in_size - kernel_size + 2 * padding) / stride) + 1
    return output


# Inherits master Module class
class SimpleCNN(torch.nn.Module):
    """ When an instance of the SimpleCNN class is created,
        we define internal functions to represent the layers of the net """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Feature extraction
        # Channels =/= input/output (channels is number of filter outputs; need to worry about FILTER dimensions)
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1,
                                     padding=1)  # --> (32, 32, 32)

        self.batch1 = torch.nn.BatchNorm2d(32)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # --> (32, 16, 16)

        # self.dropOut = torch.nn.Dropout(p=0.5)

        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1,
                                     padding=1)  # --> (64, 16, 16)

        self.batch2 = torch.nn.BatchNorm2d(64)

        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # --> (64, 8, 8)

        # self.dropOut = torch.nn.Dropout(p=0.5)

        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,
                                     padding=1)  # --> (128, 8, 8)

        self.batch3 = torch.nn.BatchNorm2d(128)

        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # --> (128, 4, 4)

        # self.dropOut = torch.nn.Dropout(p=0.5)

        # Classification -- 18 in channels with 16 x 16 pixel-sized images = 4608 input nodes
        # 64 output nodes
        self.fc1 = torch.nn.Linear(128 * 4 * 4, 64)

        self.batch4 = torch.nn.BatchNorm1d(64)

        self.fc2 = torch.nn.Linear(64, 10)

        self.batch5 = torch.nn.BatchNorm1d(10)

        # 10 inputs = 10 outputs
        self.output = torch.nn.Softmax(dim=1)  # Usually use dim = 1; means across (row)

    def forward(self, x):
        # Pass through convolutional layer 1
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.batch3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Need to flatten data from 18 channels to 4608 input nodes, i.e. (18, 16, 16) --> (1, 4608)
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc1(x)
        x = self.batch4(x)

        x = self.fc2(x)
        x = self.batch5(x)
        x = self.output(x)
        return x


def get_train_loader(batch_size, train_set, train_sampler):
    """ Gets a batch sample of training data
        Called in the training method (for loop) """
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=4)

    return train_loader


# Why does val loader have a constant batch size?
def get_val_loader(test_set, val_sampler):
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=128, sampler=val_sampler, num_workers=4)
    return val_loader


# Why does test loader have a constant batch size?
def get_test_loader(test_set, test_sampler):
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, sampler=test_sampler, num_workers=4)
    return test_loader


def trainCNN(net, device, batch_size, n_epochs, learning_rate, train_set, test_set, train_sampler, val_sampler):
    """ The full training process (i.e. not just one iteration) """

    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("-" * 30)

    train_loader = get_train_loader(batch_size, train_set, train_sampler)
    n_batches = len(train_loader)  # Train set size / batch_size

    # Creating cost and optimizer
    loss = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    training_start_time = time.time()

    val_loss_history = []
    val_acc_history = []
    train_loss_history = []
    train_acc_history = []

    # Actual training begins
    for epoch in range(n_epochs):
        print('Epoch {}/{}'.format(epoch + 1, n_epochs))
        print('-' * 10)

        net.train()  # Dictates dropout, batchnorm, etc. behavior
        training_loss = 0.0
        training_corrects = 0
        running_loss = 0.0  # Holds loss across each epoch
        print_every = n_batches // 10
        start_time = time.time()

        # Gets training data in BATCHES
        for i, data in enumerate(train_loader, 0):

            optimizer.zero_grad()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # No longer need to wrap Tensor in Variable Class
            # Tensors now support autograd (i.e. backward()) and store gradient values
            # inputs, labels = Variable(inputs), Variable(labels)

            # Batch output -- output(S)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.detach(), dim=1)
            training_corrects += (predicted == labels).double().sum().item()
            loss_size = loss(outputs, labels)  # Sum across individual losses
            loss_size.backward()  # Backprop
            optimizer.step()  # Adam step
            running_loss += loss_size

            # Print stats after every 10 mini-batches
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                    epoch + 1, int(100 * (i + 1) / n_batches), running_loss / print_every, time.time() - start_time))
                # Reset running loss and time
                training_loss += running_loss  # Add loss to training_loss for epoch
                running_loss = 0.0
                start_time = time.time()

        average_train_loss = training_loss / n_batches
        average_train_acc = 100 * training_corrects / (n_batches * batch_size)

        print("Train loss = {:.2f}".format(average_train_loss))
        print("Train accuracy = % d %%" % average_train_acc)

        # Passing through validation
        net.eval()
        epoch_val_loss = 0  # Validation loss for the epoch
        val_loader = get_val_loader(test_set, val_sampler)
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                val_outputs = net(inputs)
                # _ is array of max values of each Tensor; predicted is array of corresponding indices (labels/argmax)
                _, predicted = torch.max(val_outputs.detach(), dim=1)
                val_loss_size = loss(val_outputs, labels)
                epoch_val_loss += val_loss_size
                val_correct += (predicted == labels).double().sum().item()
                val_total += labels.size(0)

            average_val_loss = epoch_val_loss / len(val_loader)
            val_accuracy = 100 * val_correct / val_total
            print("Validation loss = {:.2f}".format(average_val_loss))
            print("Validation accuracy = % d %%" % val_accuracy)

        train_loss_history.append(average_train_loss)
        train_acc_history.append(average_train_acc)
        val_loss_history.append(average_val_loss)
        val_acc_history.append(val_accuracy)

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))

    return train_loss_history, train_acc_history, val_loss_history, val_acc_history


def test(net, device, test_set, test_sampler):
    correct = 0
    total = 0
    test_loader = get_test_loader(test_set, test_sampler)
    net.eval()
    with torch.no_grad():
        # For each testing mini-batch
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            # _ is array of max values of each Tensor; predicted is array of corresponding indices (labels/argmax)
            _, predicted = torch.max(outputs.detach(), dim=1)
            correct += (predicted == labels).double().sum().item()  # Double corrects for testing batch size
            total += labels.size(0)  # labels is a Tensor with dimension [N,1], where N is batch sample size

    print('Testing accuracy = % d %%' % (100 * correct / total))


def run_simple_CNN():
    """ Training and testing simple CNN architecture """
    [train_set, test_set, train_sampler, val_sampler, test_sampler] = pre_processing_and_samples()
    CNN = SimpleCNN()
    # Using GPU for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    if torch.cuda.is_available():
        print("cuda is available")
        CNN.to(device)

    # Multiple GPUs
    if torch.cuda.device_count() > 1:
        CNN = nn.DataParallel(CNN)

    num_epochs = 100

    train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = \
        trainCNN(net=CNN, device=device, batch_size=64, n_epochs=num_epochs, learning_rate=0.001,
                 train_set=train_set, test_set=test_set, train_sampler=train_sampler, val_sampler=val_sampler)
    test(net=CNN, device=device, test_set=test_set, test_sampler=test_sampler)

    fig, (ax1, ax2) = plt.subplots(2)

    ax1.set_title("Loss vs. Number of Training Epochs")
    ax1.set(xlabel="Training Epoch", ylabel="Loss")
    ax1.plot(range(1, len(train_loss_hist) + 1), train_loss_hist, label="Training")
    ax1.plot(range(1, len(val_loss_hist) + 1), val_loss_hist, label="Validation")
    print(np.concatenate((train_loss_hist, val_loss_hist)))
    print(np.amax(np.concatenate((train_loss_hist, val_loss_hist))))
    ax1.set_ylim(
        (0, 1.25 * np.amax(np.concatenate((train_loss_hist, val_loss_hist), axis=0, out=None)).detach().cpu()))
    ax1.set_xticks(np.arange(1, num_epochs + 1, 1.0))
    ax1.legend()

    ax2.set_title("Accuracy vs. Number of Training Epochs")
    ax2.set(xlabel="Training Epoch", ylabel="Accuracy")
    ax2.plot(range(1, num_epochs + 1), train_acc_hist, label="Training")
    ax2.plot(range(1, num_epochs + 1), val_acc_hist, label="Validation")
    ax2.set_ylim(0, 100)  # Sets y bounds
    ax2.set_xticks(np.arange(1, num_epochs + 1, 1.0))
    ax2.legend()

    plt.tight_layout()  # Call after plotting all subplots
    plt.savefig('basic_cifar_10.png')


if __name__ == "__main__":
    run_simple_CNN()
