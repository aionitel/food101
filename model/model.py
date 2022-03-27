import torch
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torchvision.models.MobileNetV2(pretrained=True)

features = model.fc.in_features

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)