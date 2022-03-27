from model import *
import time

def train_model(model=model, criterion=criterion, optimizer=optimizer, scheduler=lr_scheduler, n_epochs=25):
    since = time.time()

    best_acc = 0.0

    for epoch in range(n_epochs):
        print('Epoch {}/{}'.format(epoch, n_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0