import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn

model = torchvision.models.MobileNetV2(pretrained=True)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)