import torch
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torchsummary import summary

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# base pytorch model
model = torchvision.models.densenet121(pretrained=True)

print(model.eval)

# freeze all layers
for param in model.parameters():
    param.requires_grad = False

model.classifer = nn.Sequential(nn.Linear(1024, 512), nn.LeakyReLU(), nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 101)) # add final trainable linear layers

print(model.eval)

# move model to device (cpu or gpu)
model = model.to(device)

# loss function
criterion = nn.CrossEntropyLoss()

# stochastic gradient descent optimizer
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# learning rate
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)