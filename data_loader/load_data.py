import torch
import torchvision
import torchvision.transforms as transform

def download_data():
    trainset = torchvision.datasets.Food101(root='./data', download=True, transform=transform)
    testset = torchvision.datasets.Food101(root='./data', train=False, download=True, transform=transform)

    return trainset, testset

def load_data(trainset, testset):
    batch_size = 5
    
    transforms = transform.Compose([
        transform.RandomHorizontalFlip(),
        transform.RandomVerticalFlip(),
        transform.RandomResizedCrop(224),
        transform.GaussianBlur(),
        transform.ToTensor(),
        transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = trainset['train'].classes

    return trainloader, testloader