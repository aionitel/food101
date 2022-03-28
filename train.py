import enum
from model import *
from data_loader import *
import time, copy

# main training function
def train_model(model=model, criterion=criterion, optimizer=optimizer, scheduler=lr_rate, n_epochs=25):
    train_data, test_data = download_data()
    print('Downloaded data')

    trainloader, testloader = load_data(train_data, test_data)
    print('Loaded and prepared data for model use')

    running_loss = 0.0

    for i, (inputs, labels) in enumerate(trainloader):
        print(i)
    
        
# root training activation func
if __name__ == '__main__':
    # trained model for future use
    foodnet = train_model()