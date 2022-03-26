import torchvision

model = torchvision.models.MobileNetV2(pretrained=True)

num_ftrs = model.fc.in_features

