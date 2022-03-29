import enum
from model import *
from data_loader import *
import time, copy

# main training function
def train_model(model=model, criterion=criterion, optimizer=optimizer, scheduler=lr_rate, n_epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    since = time.time() # start timer

    model.to(device) # move model to device

    train_data, test_data = download_data()
    print('Downloaded data')

    trainloader, testloader = load_data(train_data, test_data)
    print('Loaded and prepared data for model use')

    running_loss = 0.0

    for epoch in range(n_epochs): # main training loop for n_epochs
        print('Epoch {}/{}'.format(epoch, n_epochs - 1))
        print('-' * 10)

        for inputs, labels in trainloader:
            # move data to device
            inputs.to(device)
            labels.to(device)

            # forward pass
            with torch.set_grad_enabled():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                
# train model
if __name__ == '__main__':
    # trained model for future use
    foodnet = train_model()