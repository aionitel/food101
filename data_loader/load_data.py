import torch
import torchvision
import torchvision.transforms as transforms

def load_data():
    batch_size = 5
    
    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.Food101(root='./data', download=True, transform=transforms)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.Food101(root='./data', train=False, download=True, transform=transforms)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = trainset['train'].classes

def load_datas():
    print('loaded data')