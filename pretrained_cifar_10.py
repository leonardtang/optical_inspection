from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torch.utils.data import SubsetRandomSampler
from torchvision import models, transforms
import matplotlib.pyplot as plt
import time
import copy

# Top level data directory. Here we assume the format of the directory conforms to the ImageFolder structure
data_dir = "./cifardata"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "alexnet"
num_classes = 10
batch_size = 32
num_epochs = 15
# True: feature extraction and False: fine tuning
feature_extract = True


def pre_processing_and_samples():
    seed = 42
    np.random.seed(seed)  # For calling np.random()
    torch.manual_seed(seed)  # For calling torch.rand()

    # Scaling UP (3 x 32 x 32 --> 3 x 224 x 224); augmenting (horizontal flip) for data robustness
    transform_train = transforms.Compose([transforms.Resize(224),
                                          transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    transform_test = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    train_set = torchvision.datasets.CIFAR10(root='./cifardata', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./cifardata', train=False, download=True, transform=transform_test)

    number_training_samples = 50000
    train_sampler = SubsetRandomSampler(np.arange(number_training_samples, dtype=np.int64))

    number_val_samples = 5000
    val_sampler = SubsetRandomSampler(
        np.arange(number_training_samples, number_training_samples + number_val_samples, dtype=np.int64))

    number_test_samples = 5000
    test_sampler = SubsetRandomSampler(np.arange(number_test_samples, dtype=np.int64))

    return [train_set, test_set, train_sampler, val_sampler, test_sampler]


def get_train_loader(batch_size, train_set, train_sampler):
    """ Gets a batch sample of training data; called during training process for each batch """
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=8)
    return train_loader


def get_val_loader(train_set, val_sampler):
    val_loader = torch.utils.data.DataLoader(train_set, batch_size=128, sampler=val_sampler, num_workers=8)
    return val_loader


def get_test_loader(test_set, test_sampler):
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, sampler=test_sampler, num_workers=8)
    return test_loader


def set_parameter_requires_grad(model, feature_extracting):
    """ Feature extraction: no need for backprop through all layers
        Finetuning: must have backprop (grad) through all layers """

    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    """ Initialize a pre-trained model """
    """ All of these final layers will have .requires_grad = True (by default) """

    model_ft = None
    input_size = 0

    if model_name == "resnet":
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)  # Set all layers to not require grad
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)  # This final layer will require grad by default
        input_size = 224

    elif model_name == "alexnet":
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        # Reshaping final output layer to match with number of classes for data set
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 Be careful, expects (299,299) sized images and has auxiliary output """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


# Initialize pre-trained model
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)


# print(model_ft)


def train_model(model, device, batch_size, learning_rate, train_set, train_sampler, val_sampler, num_epochs=25,
                is_inception=False):
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", num_epochs)
    print("learning_rate=", learning_rate)
    print("-" * 30)

    train_loader = get_train_loader(batch_size, train_set, train_sampler)
    n_batches = len(train_loader)
    print_every = n_batches // 10
    val_loader = get_val_loader(train_set, val_sampler)

    # Default (if fine tuning): update all parameters
    params_to_update = model.parameters()
    print("Params to learn:")
    # Only update the last parameter if performing feature extraction
    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad:
                print("\t", name)

    since = time.time()
    val_loss_history = []
    val_acc_history = []
    train_loss_history = []
    train_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Handles optimizing different layers
    optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # DataLoader requires CPU Tensors, now can switch to GPU Tensors
        model.train()
        batch_loss = 0.0
        training_loss = 0.0
        training_corrects = 0

        # Training
        for i, data in enumerate(train_loader, 0):
            # Special case for Inception V3
            if is_inception:
                outputs, aux_outputs = model(inputs)
                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux_outputs, labels)
                loss = loss1 + 0.4 * loss2

            else:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                _, predicted = torch.max(outputs.detach(), dim=1)
                loss = criterion(outputs, labels)  # Loss size is Tensor object

            batch_loss += loss
            training_corrects += (predicted == labels).double().sum().item()

            # Backprop
            loss.backward()  # Only gets called on
            optimizer.step()

            # Print stats after every 10 mini-batches
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t batch_loss: {:.2f} took: {:.2f}s".format(
                    epoch + 1, int(100 * (i + 1) / n_batches), batch_loss / print_every, time.time() - since))
                # Reset running loss and time
                training_loss += batch_loss
                batch_loss = 0.0

        average_train_loss = training_loss / n_batches
        print("Average Training Loss: {:.2f}".format(average_train_loss))
        average_train_accuracy = training_corrects / (n_batches * batch_size)
        print("Training Accuracy: {:.2f}".format(average_train_accuracy))
        train_loss_history.append(average_train_loss)
        train_acc_history.append(average_train_accuracy)

        # Validation
        model.eval()
        epoch_val_loss = 0  # Validation loss for the epoch
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                val_outputs = model(inputs)
                _, predicted = torch.max(val_outputs.detach(), dim=1)
                val_loss_size = criterion(val_outputs, labels)
                epoch_val_loss += val_loss_size
                val_correct += (predicted == labels).double().sum().item()
                val_total += labels.size(0)

            average_val_loss = epoch_val_loss / len(val_loader)
            val_accuracy = 100 * val_correct / val_total
            print("Average validation loss = {:.2f}".format(average_val_loss))
            print("Validation accuracy = % d %%" % val_accuracy)

        # Deep copy the best model (so far)
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())

        # Keep track of stats for plotting

        val_loss_history.append(epoch_val_loss)
        val_acc_history.append(val_accuracy)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Best model weights loaded for testing
    model.load_state_dict(best_model_wts)
    return model, train_loss_history, train_acc_history, val_loss_history, val_acc_history


def test(model, device, test_set, test_sampler):
    correct = 0
    total = 0
    test_loader = get_test_loader(test_set, test_sampler)
    model.eval()
    with torch.no_grad():
        # For each testing mini-batch
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.detach(), dim=1)
            correct += (predicted == labels).double().sum().item()  # Double corrects for testing batch size
            total += labels.size(0)  # labels is a Tensor with dimension [N,1], where N is batch sample size

    print('Testing accuracy = % d %%' % (100 * correct / total))


def run_NN():
    """ Training and testing simple CNN architecture """
    [train_set, test_set, train_sampler, val_sampler, test_sampler] = pre_processing_and_samples()
    # Using GPU for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = initialize_model(model_name=model_name, num_classes=num_classes, feature_extract=True)[0]
    if torch.cuda.is_available():
        print("cuda is available")

        # Multiple GPUs
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            print("Model wrapped in Data Parallel")

    model.to(device)

    # Gives the best model (best weights)
    model, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = train_model(model=model, device=device,
                                                                                      batch_size=32, learning_rate=0.01,
                                                                                      train_set=train_set,
                                                                                      train_sampler=train_sampler,
                                                                                      val_sampler=val_sampler,
                                                                                      num_epochs=num_epochs)

    print(model)
    print(train_loss_hist)
    print(train_acc_hist)
    print(val_loss_hist)
    print(val_acc_hist)

    # USE VISDOM HERE INSTEAD
    # # Plot performance over training epochs
    # plt.title("Validation Accuracy vs. Number of Training Epochs")
    # plt.xlabel("Training Epochs")
    # plt.ylabel("Validation Accuracy")
    # plt.plot(range(1, num_epochs + 1), train_loss_hist, label="Pretrained")
    # plt.ylim((0, 1.))
    # plt.xticks(np.arange(1, num_epochs + 1, 1.0))
    # plt.legend()
    # plt.show()

    test(model=model, device=device, test_set=test_set, test_sampler=test_sampler)


if __name__ == "__main__":
    run_NN()
