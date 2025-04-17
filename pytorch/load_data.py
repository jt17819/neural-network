import numpy as np
from torch import from_numpy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

class CustomNumpyDataset(Dataset):
    def __init__(self, numpy_list, transform=None):
        self.numpy_list = numpy_list
        self.transform = transform

    def __len__(self):
        return len(self.numpy_list)

    def __getitem__(self, idx):
        sample = self.numpy_list[idx]
        sample = from_numpy(sample)
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    

def get_dataloaders(batch_size=128):
    my_transforms = transforms.Compose([
        transforms.ToTensor(), 
        # transforms.RandomRotation(degrees=40), 
        # transforms.Normalize(mean=[0.5], std=[0.5])  # (0.1307,), (0.3081,)
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])
    
    train_X = np.load("Data Set/Digits/Processed Training Images.npy")
    train_y = np.load("Data Set/Digits/Processed Training Labels.npy")

    train_X = train_X.reshape(train_X.shape[0], 1, 28*28)
    train_X = train_X.astype('float32')
    train_X /= 255
    num_classes = 10
    train_y = np.eye(num_classes)[train_y]
    train_X = from_numpy(train_X)
    train_y = from_numpy(train_y)
    # dataset = CustomNumpyDataset(train_X, transform=my_transforms)
    train_dataloader = DataLoader(list(zip(train_X, train_y)), batch_size=batch_size, shuffle=True)

    
    test_X = np.load("Data Set/Digits/Processed Test Images.npy")
    test_y = np.load("Data Set/Digits/Processed Test Labels.npy")
    test_X = test_X.reshape(test_X.shape[0], 1, 28*28)
    test_X = test_X.astype('float32')
    test_X /= 255
    test_y = np.eye(num_classes)[test_y]
    test_dataloader = DataLoader(list(zip(test_X, test_y)), batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader


def get_loaders(batch_size=128):
    my_transforms = transforms.Compose([
        transforms.RandomRotation(degrees=40), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.5], std=[0.5])  # (0.1307,), (0.3081,)
        # transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])

    train_dataset = datasets.MNIST('./mnist_data', train=True, download=True, transform=my_transforms)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST('./mnist_data', train=False, transform=my_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader