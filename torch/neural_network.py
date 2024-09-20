from torch import nn

class MNIST_NN_Linear(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784,100),
            nn.LeakyReLU(),
            
            nn.Linear(100,50),
            nn.LeakyReLU(),
            
            nn.Linear(50,10),
            nn.Softmax(dim=1)
        )
        
    def forward(self, X):
        X = X.view(X.shape[0], -1)
        X = self.layers(X)
        return X
    

class MNIST_CNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, stride=2),
            
            nn.Conv2d(6, 16, 5, padding=0),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, stride=2),
            
            nn.Flatten(),
            nn.Linear(256, 100),
            nn.LeakyReLU(),
            
            nn.Linear(100, 50),
            nn.LeakyReLU(),
            
            nn.Linear(50, 10),
            nn.Softmax(dim=1)
        )
        
    def forward(self, X):
        X = self.layers(X)
        return X
    
