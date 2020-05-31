import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd

data_dir = "data/"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
# model_name = "alexnet"
model_name = "vgg"

# Number of classes in the dataset
num_classes = 11

# Batch size for training (change depending on how much memory you have)
batch_size = 2

# Number of epochs to train for
num_epochs = 50

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

#use pretrained models
pre_trained=True

def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs=25, is_inception=False):
    print("Model name: " + model_name)
    since = time.time()

    test_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    iter = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        time_current = time.time() - since
        print('Consumed Time {:.0f}m {:.0f}s'.format(time_current // 60, time_current % 60))

        # Iterate over data.
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            labels = torch.squeeze(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                # Get model outputs and calculate loss
                # Special case for inception because in training it has an auxiliary output. In train
                #   mode we calculate the loss by summing the final output and the auxiliary output
                #   but in testing we only consider the final output.
                if is_inception:
                    outputs, aux_outputs = model(inputs)
                    loss1 = criterion(outputs, labels)
                    loss2 = criterion(aux_outputs, labels)
                    loss = loss1 + 0.4*loss2
                else:
                    inputs = inputs.permute(0, 2, 1, 3)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            iter += 1
            if (iter % 25 == 0):
                print("Training... iteration {}".format(iter))

            if iter % 50 == 0:
                correct = 0
                total = 0
                test_iteration = 0

                for inputs, labels in test_dataloader:
                    inputs = inputs.to(device, dtype=torch.float)
                    labels = labels.to(device, dtype=torch.long)
                    labels = torch.squeeze(labels)

                    inputs = inputs.permute(0, 2, 1, 3)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()

                    test_iteration += 1
                    if test_iteration % 500 == 0:
                        print("      Testing... Epoch: {}, Training iteration: {}, Test iteration: {}".format(epoch, iter, test_iteration))

                accuracy = 100 * correct // total
                print('   Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))

                if accuracy > best_acc:
                    print("In Epoch: {}, Iteration: {}, Accuracy: {}, Better accuracy appears!!!".format(epoch, iter, accuracy))
                    best_acc = accuracy
                    best_model = copy.deepcopy(model)
                    test_acc_history.append(best_acc.item())
                    # best_model_wts = copy.deepcopy(model.state_dict())

    torch.save(best_model, "trained_model/"+model_name+"_pretrained.pth")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    with open('log/' + model_name +'_test_accuracy_history.npy', 'wb') as f:
        np.save(f, test_acc_history)

    # load best model weights
    # model.load_state_dict(best_model_wts)
    #model = torch.load(best_model)
    return test_acc_history

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=pre_trained)

# Print the model we just instantiated
print(model_ft)

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

class ActivitySkeletalDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, key, model_name, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_hdf(root_dir+"data_training_test.h5", key=key+"_data")
        self.labels = pd.read_hdf(root_dir+"data_training_test.h5", key=key+"_label")
        self.root_dir = root_dir
        # self.transform = transform
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.data.iloc[idx, 0:]

        # if model_name == 'resnet':
        if model_name == 'resnet':
            data = np.array([data, data, data])
            data = data.reshape(3, 18, 2)
            data = data.astype('double').reshape(3, -1, 2)

        if model_name == 'alexnet':
            data = np.array(data).reshape(-1, 1)
            data_column_expanded = data
            for i in range(8):
                data_column_expanded = np.concatenate((data_column_expanded, data_column_expanded), axis=1)
            data_column_expanded = data_column_expanded[:, 0:252]
            data_row_column_expanded = data_column_expanded
            for i in range(6):
                data_row_column_expanded = np.concatenate((data_row_column_expanded, data_column_expanded), axis=0)
            data = np.array([data_row_column_expanded, data_row_column_expanded, data_row_column_expanded]).astype('double')

        if model_name == 'vgg':
            data = np.array(data).reshape(-1, 1)
            data_column_expanded = data
            for i in range(8):
                data_column_expanded = np.concatenate((data_column_expanded, data_column_expanded), axis=1)
            data_column_expanded = data_column_expanded[:, 0:252]
            data_row_column_expanded = data_column_expanded
            for i in range(6):
                data_row_column_expanded = np.concatenate((data_row_column_expanded, data_column_expanded), axis=0)
            data = np.array([data_row_column_expanded, data_row_column_expanded, data_row_column_expanded]).astype(
                'double')


        result = self.labels.iloc[idx]
        result = np.array([result])
        result = result.astype('int')

        if self.transform:
            a = data
            data = self.transform(data)
            result = self.transform(result)

        return data, result

# Create training and test datasets
train_activity_dataset = ActivitySkeletalDataset(data_dir, 'train', 'alexnet', data_transforms['train'])
test_activity_dataset = ActivitySkeletalDataset(data_dir, 'test', 'alexnet', data_transforms['test'])

# Create training and test dataloaders
train_dataloader = torch.utils.data.DataLoader(train_activity_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataloader = torch.utils.data.DataLoader(test_activity_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

criterion = nn.CrossEntropyLoss()

# Train and evaluate
# model_ft, hist = train_model(model_ft, train_dataloader, test_dataloader, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))
hist = train_model(model_ft, train_dataloader, test_dataloader, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))