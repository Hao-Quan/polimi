from os import listdir
from os.path import isfile, join

import pandas as pd
import os

root_dir = "./"

header = []
for i in range(18):
    header.append("x_" + str(i))
    header.append("y_" + str(i))

directory_list_carosello = ["1_carosello", "3_carosello", "4_carosello", "9_carosello", "10_carosello"]
str_x = "/X.txt"
str_y = "/Y.txt"
landmarks_frame_X = pd.DataFrame(columns=header)
label_Y = pd.DataFrame(columns=["Y"])

for i in directory_list_carosello:
    path_current_carosello = root_dir + i
    p = os.walk(path_current_carosello)
    onlyfiles = [f for f in listdir(path_current_carosello) if isfile(join(path_current_carosello, f))]
    # (_, _, filenames) = os.walk(path_curent_carosello).next()
    tracese_name = []
    for (dirpath, dirnames, filenames) in os.walk(path_current_carosello):
        tracese_name.append(dirpath)
    tracese_name.sort()
    tracese_name.pop(0)

    for trace in tracese_name:
        print("Processing: " + trace + str_x)
        current_file_data_X = pd.read_csv(trace+str_x, sep=',', names=header)
        current_file_data_Y = pd.read_csv(trace + str_y, sep=',', names="Y")
        landmarks_frame_X = landmarks_frame_X.append(current_file_data_X)
        label_Y = label_Y.append(current_file_data_Y)

print("")