import torch
import pandas as pd
import numpy as np

from datetime import datetime



print(torch.__version__)

a = np.load("log/densenet_test_accuracy_history.npy")
print("x")

model_name = "resnet"
test_acc_history = np.array([0])

with open('log/' + model_name + '/' + datetime.today().strftime('%Y-%m-%d') + "_" + model_name + '_test_accuracy_history.npy', 'wb') as f:
    np.save(f, test_acc_history)
# from torchvision import _C
# if hasattr(_C, 'CUDA_VERSION'):
#     cuda = _C.CUDA_VERSION


# def hello_world():
#     print("hello world")
# if __name__ == '__main__':
#     import pdb; pdb.set_trace()
#     hello_world()



