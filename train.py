from model import *
from data_loader import *
import time, copy

# main training function
def train_model(model=model, criterion=criterion, optimizer=optimizer, scheduler=lr_scheduler, n_epochs=25):
    trainset, testset = download_data()
    print('Downloaded data')

    trainloader, testloader = load_data(trainset, testset)
    print('Loaded and prepared data for model use')

    since = time.time()

    best_acc = 0.0

    # main training loop
    for epoch in range(n_epochs):
        print('Epoch {}/{}'.format(epoch, n_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            running_loss = 0.0
            running_corrects = 0

            # training epoch
            if phase == 'train':
                # set training mode
                model.train()
                
                for inputs, labels in trainloader:
                    inputs.to(device)
                    labels.to(device)

                    # zero parameter gradients
                    optimizer.zero_grad()

                    # forward pass
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)

                        loss = criterion(outputs, labels)

                        # backward pass
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / trainset.__len__()
            epoch_acc = running_corrects.double() / trainset.__len__()

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
        
# root training activation func
if __name__ == '__main__':
    # trained model for future use
    food_model = train_model()