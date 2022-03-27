from model import *
from data_loader import *
import time

# main training function
def train_model(model=model, criterion=criterion, optimizer=optimizer, scheduler=lr_scheduler, n_epochs=25):
    trainset, testset = download_data()
    trainloader, testloader = load_data(trainset, testset)

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
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
        
# root training activation func
if __name__ == '__main__':
    # trained model for future use
    food_model = train_model()