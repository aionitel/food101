import enum
from model import *
from data_loader import *
import time, copy
from torchvision.transforms import ToTensor

# main training function
def train_model(model=model, criterion=criterion, optimizer=optimizer, scheduler=lr_rate, n_epochs=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    since = time.time() # start timer

    model.to(device) # move model to device

    train_data, test_data = download_data()
    print('Downloaded data')

    trainloader, testloader = load_data(train_data, test_data)
    print('Loaded and prepared data for model use')

    running_loss = 0.0

    for epoch in range(n_epochs): # main training loop for n_epochs
        print(f'Epoch {epoch}/{n_epochs}')
        print('-' * 10)

        for inputs, labels in trainloader:
            # move data to device
            inputs.to(device)
            labels.to(device)

            # zero gradients
            optimizer.zero_grad()

            #forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if epoch % 10 == 9:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
                
# train model
if __name__ == '__main__':
    # trained model for future use
    foodnet = train_model()