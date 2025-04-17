from model import CNN, LNN
import numpy as np
import torch
from PIL import Image
from train import train, test
from torch.utils.data import Dataset, DataLoader


train_X = np.load("Data Set/Digits/Processed Training Images.npy")
train_y = np.load("Data Set/Digits/Processed Training Labels.npy")
test_X = np.load("Data Set/Digits/Processed Test Images.npy")
test_y = np.load("Data Set/Digits/Processed Test Labels.npy")
train_X = train_X.reshape(train_X.shape[0], 1, 28*28)
train_X = train_X.astype('float32')
train_X /= 255
num_classes = 10
train_y = np.eye(num_classes)[train_y]

test_X = test_X.reshape(test_X.shape[0], 1, 28*28)
test_X = test_X.astype('float32')
test_X /= 255
# test_X  = torch.FloatTensor(test_X)
correct = 0

device = torch.device('cpu')
print(device)


# new_model = CNN().to(device)
# new_model.load_state_dict(torch.load("mnist-pytorch.pt"))
# print(new_model.train())
# print(new_model.eval())
# img = test_X.data[0]
# img = test_X[0]
# img_pil = Image.fromarray(img)
# img = np.array(img_pil.resize((28, 28), Image.Resampling.LANCZOS))
# img = np.invert([img])
# img = img.reshape(img.shape[0], 1, 28*28)
# img = img.astype('float32')
# img /= 255
# print(test_X.data[0].to(device))
# print(new_model(test_X[0]))num_epochs = 2
train_X = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_y)
test_X = torch.from_numpy(test_X)
test_y = torch.from_numpy(test_y)


train_loader = DataLoader(list(zip(train_X,train_y)), shuffle=True, batch_size=128)
test_loader = DataLoader(list(zip(test_X,test_y)))

num_epochs = 2
model = LNN().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
torch.save(model.state_dict(), "np-mnist-pytorch.pt")
