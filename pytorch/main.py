import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import CNN, Model
from load_data import get_dataloaders
from train import train, test

def main():
    device = torch.device('cpu')
    print(device)

    train_loader, test_loader = get_dataloaders(batch_size=100)

    model = Model().to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # cnn = CNN()
    # cost_func = nn.CrossEntropyLoss()
    num_epochs = 200

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.5)
    
    for epoch in range(1, num_epochs + 1):
        loss = train(model, device, train_loader, optimizer, criterion)
        
        if epoch % 10 == 0:
            print(f'Epoch: {epoch} and loss: {loss}')
        
        if epoch % 10 == 0:
            test_loss, correct = test(model, device, test_loader)
            test_loss /= len(test_loader.dataset)

            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        
    torch.save(model.state_dict(), "mnist-pytorch.pt")

if __name__ == "__main__":
    main()
