import torch.nn as nn
 

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 320)
        x = self.fc_layers(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(784, 200),
            # nn.ReLU(),
            nn.Tanh(),
            # nn.Dropout(),
            nn.Linear(200,100),
            # nn.ReLU(),
            nn.Tanh(),
            # nn.Linear(100,50),
            nn.Linear(100,10),
            nn.Softmax(dim=2)
        )
    
    def forward(self, x):
        x = self.fc_layers(x)
        return x