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
batch_size = 64

class_mapping = {"sitting_phone_talking": 10, "standing": 3, "walking_phone": 6, "walking_cart": 9,
                 "walking_fast": 2, "wandering": 7, "walking_slow": 1, "standing_phone_talking": 4,
                 "sitting": 0, "window_shopping": 5, "walking_phone_talking": 8}


activity_class_name = "walking_cart"
activity_class_number = 9

def test_model(model, test_dataloader, is_inception=False):
    correct = 0
    correct_semantics = 0
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

        idx_sample_list = [i for i in range(len(labels)) if labels[i] == 9]
        inputs_cpu = inputs.cpu()
        labels_cpu = labels.cpu()

        predicted_semantics = predicted

        for i in range(len(idx_sample_list)):
            if predicted[idx_sample_list[i]] != activity_class_number:
                axis_x = np.squeeze(np.array(inputs_cpu[i]))[0]
                axis_y = np.squeeze(np.array(inputs_cpu[i]))[1]

                missed_joints_boolean = axis_x == -1
                missed_joints = [i for i, x in enumerate(missed_joints_boolean) if x]

                # # Plot - Start
                # n = [i for i in range(18)]
                # mask = axis_x != -1
                #
                # fig, ax = plt.subplots()
                # ax.scatter(axis_x[mask], axis_y[mask])
                #
                # for j, txt in enumerate(n):
                #     ax.annotate(txt, (axis_x[j], axis_y[j]))
                #
                # plt.title("Label: " + activity_class_name + " (" + str(
                #     activity_class_number) + "), Predicted: " + list(class_mapping.keys())[
                #               list(class_mapping.values()).index(predicted[i].item())] + " (" + str(
                #     predicted[i].item()) + ")\nMissed joints: " + " ".join(str(x) for x in missed_joints))
                # plt.gca().invert_xaxis()
                # plt.gca().invert_yaxis()
                # # plt.savefig("img/study_case_9/" + str(k) + ".png")
                # #plt.show()
                # # Plot - End

                if 3 not in missed_joints and not 4 in missed_joints and not 6 in missed_joints and not 7 in missed_joints \
                    and 8 not in missed_joints and not 11 in missed_joints:
                    # k == 11 (test image case)
                    plt.show()

                    x_8 = axis_x[8]
                    y_8 = axis_y[8]
                    x_11 = axis_x[11]
                    y_11 = axis_y[11]
                    x_min_8_11 = np.amin([x_8, x_11])
                    y_min_8_11 = np.amin([y_8, y_11])
                    x_max_8_11 = np.amax([x_8, x_11])
                    y_max_8_11 = np.amax([y_8, y_11])

                    x_3 = axis_x[3]
                    y_3 = axis_y[3]
                    x_4 = axis_x[4]
                    y_4 = axis_y[4]
                    x_6 = axis_x[6]
                    y_6 = axis_y[6]
                    x_7 = axis_x[7]
                    y_7 = axis_y[7]
                    ordered_x_3_4_6_7 = [x_3, x_4, x_6, x_7]
                    ordered_y_3_4_6_7 = [y_3, y_4, y_6, y_7]
                    ordered_x_3_4_6_7.sort()
                    ordered_y_3_4_6_7.sort()
                    x_min_3_4_6_7 = np.amin([x_3, x_4, x_6, x_7])
                    y_min_3_4_6_7 = np.amin([y_3, y_4, y_6, y_7])
                    x_max_3_4_6_7 = np.amax([x_3, x_4, x_6, x_7])
                    y_max_3_4_6_7 = np.amax([y_3, y_4, y_6, y_7])

                    x_second_smallest_3_4_6_7 = ordered_x_3_4_6_7[1]
                    x_second_largest_3_4_6_7 = ordered_x_3_4_6_7[2]
                    y_second_smallest_3_4_6_7 = ordered_y_3_4_6_7[1]
                    y_second_largest_3_4_6_7 = ordered_y_3_4_6_7[2]

                    if y_max_3_4_6_7 < y_min_8_11:
                        if x_min_3_4_6_7 > x_min_8_11 or x_second_smallest_3_4_6_7 > x_min_8_11:
                            if x_second_largest_3_4_6_7 - x_max_8_11 > 15 and x_max_3_4_6_7 - x_max_8_11 > 20:
                                # We can predict is "walking_cart" - hand on left
                                predicted_semantics[idx_sample_list[i]] = 9

                        elif x_max_3_4_6_7 < x_min_8_11 or x_second_smallest_3_4_6_7 < x_min_8_11:
                            if x_min_8_11 - x_second_smallest_3_4_6_7 > 15 and x_min_8_11 - x_min_3_4_6_7 > 20:
                                # We can predict is "walking_cart" - hand on right
                                predicted_semantics[idx_sample_list[i]] = 9
                        o = 1


                    o = 1

                k += 1

        correct_semantics += (predicted_semantics == labels).sum()

        m = 1




    accuracy = 100 * correct / total
    print('Accuracy: {:.2f}'.format(accuracy))

    accuracy_semantics = 100 * correct_semantics / total
    print('Accuracy Semantics: {:.2f}'.format(accuracy_semantics))

    print("Correct {} sample with label 9 'walking_cart'".format(correct_semantics - correct))

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

# #START Use train dataset
#
# train_activity_dataset = ActivitySkeletalDataset(data_dir, 'train', data_transforms['train'])
# #test_activity_dataset = ActivitySkeletalDataset(data_dir, 'test', data_transforms['test'])
#
# # Create training and test dataloaders
# train_dataloader = torch.utils.data.DataLoader(train_activity_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
#
# # Detect if we have a GPU available
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# model_ft = model_ft.to(device)
#
# # Train and evaluate
# test_model(model_ft, train_dataloader)
#
# # END Use train dataset


# Create training and test datasets
#train_activity_dataset = ActivitySkeletalDataset(data_dir, 'train', 'resnet', data_transforms['train'])
test_activity_dataset = ActivitySkeletalDataset(data_dir, 'test', data_transforms['test'])

# Create training and test dataloaders
#train_dataloader = torch.utils.data.DataLoader(train_activity_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataloader = torch.utils.data.DataLoader(test_activity_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = model_ft.to(device)

# Train and evaluate
test_model(model_ft, test_dataloader, is_inception=(model_name=="inception"))



