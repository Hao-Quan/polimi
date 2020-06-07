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
from datetime import datetime

data_dir = "data/"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
# model_name = "densenet"
# model_path = "trained_model/densenet_pretrained.pth"

model_name = "resnet"
model_path = "trained_model/2020-06-02_bs64_withoutpad_resnet_pretrained.pth"


# Batch size for training (change depending on how much memory you have)
batch_size = 32

activity_class_name = ""

def test_model(model, test_dataloader, is_inception=False):
    correct = 0
    total = 0

    # 9 is walking_cart
    label_to_verify = 9
    k = 0
    for inputs, labels in test_dataloader:
        #inputs = inputs.to(device, dtype=torch.float)
        #labels = labels.to(device, dtype=torch.long)
        labels = torch.squeeze(labels)

        inputs = inputs.permute(0, 2, 1, 3)

        # Use pretrained model to compute predict label
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

        sample_list = [i for i in range(len(labels)) if labels[i] == 9]
        inputs_cpu = inputs.cpu()
        labels_cpu = labels.cpu()

        for i in range(len(sample_list)):
            if predicted[sample_list[i]] != 9:
                activity_class_name = "walking_cart"
                activity_class_number = 9
                axis_x = np.squeeze(np.array(inputs_cpu[i]))[0]
                axis_y = np.squeeze(np.array(inputs_cpu[i]))[1]

                missed_joints_boolean = axis_x == -1
                missed_joints = [i for i, x in enumerate(missed_joints_boolean) if x]
                n = [i for i in range(18)]
                mask = axis_x != -1

                fig, ax = plt.subplots()
                ax.scatter(axis_x[mask], axis_y[mask])

                for j, txt in enumerate(n):
                    ax.annotate(txt, (axis_x[j], axis_y[j]))

                # plt.plot(axis_x, axis_y, 'ro')
                plt.title(activity_class_name + " (" + str(
                    activity_class_number) + "), " + "Missed joints: " + " ".join(str(x) for x in missed_joints))
                plt.gca().invert_xaxis()
                plt.gca().invert_yaxis()
                plt.savefig("img/study_case_9/" + str(k) + ".png")
                k += 1
                m = 2

        m = 1

        # # Use semantics to compute predict label
        # for i in range(len(labels)):
        #     if labels[i].item() == 9:
        #         inputs = inputs.cpu()
        #         labels = labels.cpu()
        #         activity_class_name = "walking_cart"
        #         activity_class_number = 9
        #         axis_x = np.squeeze(np.array(inputs[i]))[0]
        #         axis_y = np.squeeze(np.array(inputs[i]))[1]
        #
        #         missed_joints_boolean = axis_x == -1
        #         missed_joints = [i for i, x in enumerate(missed_joints_boolean) if x]
        #         n = [i for i in range(18)]
        #         mask = axis_x != -1
        #
        #         fig, ax = plt.subplots()
        #         ax.scatter(axis_x[mask], axis_y[mask])
        #
        #         for i, txt in enumerate(n):
        #             ax.annotate(txt, (axis_x[i], axis_y[i]))
        #
        #         # plt.plot(axis_x, axis_y, 'ro')
        #         plt.title(activity_class_name + " (" + str(
        #             activity_class_number) + "), " + "Missed joints: " + " ".join(str(x) for x in missed_joints))
        #         plt.gca().invert_xaxis()
        #         plt.gca().invert_yaxis()
        #         plt.show()
        #         k=2




    accuracy = 100 * correct / total
    print('Accuracy: {}'.format(accuracy))

    return accuracy

model_ft = torch.load(model_path)

# Print the model we just instantiated
print(model_ft)

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

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

        if model_name == 'resnet':
            # 2D Version
            data = np.array([data])
            vector_x = []
            vector_y = []
            for i in range(0, 36):
                if i % 2 == 0:
                    vector_x.append(data[0][i])
                else:
                    vector_y.append(data[0][i])
            data = np.stack([np.array(vector_x), np.array(vector_y)])
            data = data.astype('double')

        if model_name == 'densenet':
            data = np.array(data).reshape(-1, 1)
            data_column_expanded = data
            for i in range(5):
                data_column_expanded = np.concatenate((data_column_expanded, data_column_expanded), axis=1)
            data = np.array([data_column_expanded, data_column_expanded, data_column_expanded]).astype(
                'double')

        result = self.labels.iloc[idx]
        result = np.array([result])
        result = result.astype('int')

        if self.transform:
            data = self.transform(data)
            result = self.transform(result)

        return data, result

# Create training and test datasets
#train_activity_dataset = ActivitySkeletalDataset(data_dir, 'train', 'resnet', data_transforms['train'])
test_activity_dataset = ActivitySkeletalDataset(data_dir, 'test', data_transforms['test'])

# Create training and test dataloaders
#train_dataloader = torch.utils.data.DataLoader(train_activity_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataloader = torch.utils.data.DataLoader(test_activity_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
model_ft = model_ft.to(device)

# Train and evaluate
#hist = train_model(model_ft, train_dataloader, test_dataloader, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))
test_model(model_ft, test_dataloader, is_inception=(model_name=="inception"))