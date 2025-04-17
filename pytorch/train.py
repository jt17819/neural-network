import torch.nn.functional as F
from torch import no_grad


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        y_pred = model.forward(data)
        loss = criterion(y_pred.squeeze(1), target)
        # loss = F.mse_loss(y_pred.squeeze(1), target.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Training with {i+1}", end="\r")
    return loss


def test(model, device, test_loader):
    test_loss = 0
    correct = 0
    model.eval()
    with no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model.forward(data)
            test_loss += F.mse_loss(output, target.float().unsqueeze(1)).item() # sum up batch 
            pred = output.max(2)[1] # get the index of the max log-
            target = target.max(1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    return test_loss, correct
