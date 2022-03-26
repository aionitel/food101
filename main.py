from data_loader import *
import torch

device = torch.device('cudda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    print(device)