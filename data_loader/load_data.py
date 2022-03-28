import torch
import torchvision
import torchvision.transforms as transform

# download base food image dataset
def download_data():
    trainset = torchvision.datasets.Food101(root='./data', download=True)
    testset = torchvision.datasets.Food101(root='./data', download=True)

    transforms = transform.Compose([
        transform.RandomHorizontalFlip(),
        transform.RandomVerticalFlip(),
        transform.RandomResizedCrop(224),
        transform.GaussianBlur(kernel_size=3),
        transform.ToTensor(),
        transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = torchvision.datasets.ImageFolder('./data/food-101/images', transform=transforms)
    test_data = torchvision.datasets.ImageFolder('./data/food-101/images', transform=transforms)

    return train_data, test_data

# load and transform dataset for model
def load_data(train_data, test_data):
    batch_size = 5
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader 