import torch
import torchvision
import torchvision.transforms as transform

# download base food image dataset
def download_data():
    trainset = torchvision.datasets.Food101(root='./data', download=True)
    testset = torchvision.datasets.Food101(root='./data', download=True)

    transforms = transform.Compose([
        transform.Resize(299),
        transform.RandomResizedCrop(299),
        transform.RandomHorizontalFlip(),
        transform.RandomVerticalFlip(),
        transform.ToTensor(),
        transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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