import torch
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# base pytorch model
model = torchvision.models.mobilenet_v2(pretrained=True)

# features
features = model.features

# move model to device (cpu or gpu)
model = model.to(device)

# loss function
criterion = nn.CrossEntropyLoss()

# stochastic gradient descent optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# learning rate
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)