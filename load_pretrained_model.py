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

input_size = 224

batch_size = 512

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

model = torch.load("model/pretrained.pth")
model.eval()

class ActivitySkeletalDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, key, transform=None):
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
        data = np.array([data, data, data])
        data = data.reshape(3, 18, 2)
        data = data.astype('double').reshape(3, -1, 2)

        result = self.labels.iloc[idx]
        result = np.array([result])
        result = result.astype('int')

        if self.transform:
            data = self.transform(data)
            result = self.transform(result)

        return data, result


# Create training and test datasets

test_activity_dataset = ActivitySkeletalDataset(data_dir, 'test', data_transforms['test'])

# Create training and test dataloaders

test_dataloader = torch.utils.data.DataLoader(test_activity_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

total = 0
correct = 0

for inputs, labels in test_dataloader:
    inputs = inputs.to(device, dtype=torch.float)
    labels = labels.to(device, dtype=torch.long)
    labels = torch.squeeze(labels)

    inputs = inputs.permute(0, 2, 1, 3)
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

accuracy = 100 * correct / total
print('Test Acc: {:4f}'.format(accuracy))