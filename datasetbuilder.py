from __future__ import print_function, division
import os
from os import listdir
from os.path import isfile, join

import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import warnings

warnings.filterwarnings("ignore")

plt.ion()

header = []
for i in range(18):
    header.append("x_" + str(i))
    header.append("y_" + str(i))

# landmarks_frame = pd.read_csv('data/1_carosello/0/X.csv',
#                               sep=',',
#                               names=header)


# landmarks_frame_test = pd.read_csv('data/test.csv',
#                               sep=',',
#                               names=["x_0", "y_0", "x_1", "y_1"])


# n = 65
# img_name = landmarks_frame.iloc[n, 0]
# landmarks = landmarks_frame.iloc[n, 1:]
# landmarks = np.asarray(landmarks)
# landmarks = landmarks.astype('float').reshape(-1, 2)

# print('Image name: {}'.format(img_name))
# print('Landmarks shape: {}'.format(landmarks.shape))
# print('First 4 Landmarks: {}'.format(landmarks[:4]))

# def show_landmarks(image, landmarks):
#     """Show image with landmarks"""
#     plt.imshow(image)
#     plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
#     plt.pause(0.001)  # pause a bit so that plots are updated
#
# plt.figure()
# show_landmarks(io.imread(os.path.join('data/faces/', img_name)),
#                landmarks)
# plt.show()

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # directory_list_carosello = ["1_carosello", "3_carosello", "4_carosello", "9_carosello", "10_carosello"]
        # str_x = "/X.txt"
        # str_y = "/Y.txt"
        # self.landmarks_frame_X = pd.DataFrame(columns=header)
        # self.label_Y = pd.DataFrame(columns=["Y"])
        #
        # for i in directory_list_carosello:
        #     path_current_carosello = root_dir + i
        #     p = os.walk(path_current_carosello)
        #     onlyfiles = [f for f in listdir(path_current_carosello) if isfile(join(path_current_carosello, f))]
        #     # (_, _, filenames) = os.walk(path_curent_carosello).next()
        #     tracese_name = []
        #     for (dirpath, dirnames, filenames) in os.walk(path_current_carosello):
        #         tracese_name.append(dirpath)
        #     tracese_name.sort()
        #     tracese_name.pop(0)
        #
        #     for trace in tracese_name:
        #         print("Processing: " + trace + str_x)
        #         current_file_data_X = pd.read_csv(trace+str_x, sep=',', names=header)
        #         current_file_data_Y = pd.read_csv(trace + str_y, sep=',', names="Y")
        #         self.landmarks_frame_X = self.landmarks_frame_X.append(current_file_data_X)
        #         self.label_Y = self.label_Y.append(current_file_data_Y)

        self.landmarks_frame_X = pd.read_hdf(root_dir+"data.h5", key="X")
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame_X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        # image = io.imread(img_name)
        landmarks = self.landmarks_frame_X.iloc[idx, 0:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = landmarks

        if self.transform:
            sample = self.transform(sample)

        return sample


face_dataset = FaceLandmarksDataset(root_dir='data/')

fig = plt.figure()

for i in range(len(face_dataset)):
    sample = face_dataset[i]

    print(i, sample.shape)

    # ax = plt.subplot(1, 4, i + 1)
    # plt.tight_layout()
    # ax.set_title('Sample #{}'.format(i))
    # ax.axis('off')
    # show_landmarks(**sample)

    ax = plt.plot(sample[:, 0], sample[:, 1], 'ro')
    #plt.tight_layout()
    #ax.set_title('Sample #{}'.format(i))
    #ax.axis('off')

    # if i == 3:
    #     plt.show()
    #     break

print("")
