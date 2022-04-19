import torch
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torchsummary import summary

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# base densenet model
model = torchvision.models.resnet18(pretrained=True)

num_ftrs = model.fc.in_features

# freeze all layers
for param in model.parameters():
    param.requires_grad = False

# add final trainable linear layers
model.fc = nn.Linear(num_ftrs, 101) # add final linear layer for 101 classes

# loss function
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

# learning rate scheduler
lr_rate = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# model summary
# summary(model, (3, 299, 299))