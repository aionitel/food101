from model import *
from data_loader import *
import time, copy

# main training function
def train_model(model=model, criterion=criterion, optimizer=optimizer, scheduler=lr_scheduler, n_epochs=25):
    trainset, testset = download_data()
    print('Downloaded data')

    trainloader, testloader = load_data(trainset, testset)
    print('Loaded and prepared data for model use')

    running_loss = 0.0

    for inputs in trainloader:
        print(inputs)
        
# root training activation func
if __name__ == '__main__':
    # trained model for future use
    train_model()