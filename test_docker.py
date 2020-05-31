import os
import torch
from torchvision import models, transforms

#torch.hub.set_dir("torch_hub")

#odel_ft = models.resnet18(pretrained=True)

import torch
print(torch.__version__)

model_ft = models.alexnet(pretrained=True)
torch.save(model_ft, "model/alexnet.pth")

model_ft = models.vgg11_bn(pretrained=True)
torch.save(model_ft, "model/vgg11_bn.pth")

model_ft = models.squeezenet1_0(pretrained=True)
torch.save(model_ft, "model/squeezenet1_0.pth")

model_ft = models.densenet121(pretrained=True)
torch.save(model_ft, "model/densenet121.pth")




print("Hello world")

#print(os.environ['$TORCH_HOME'])