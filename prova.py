import torch

x = torch.tensor([1, 2, 3, 4])
y = torch.unsqueeze(x, 0)
z = torch.unsqueeze(x, 1)
