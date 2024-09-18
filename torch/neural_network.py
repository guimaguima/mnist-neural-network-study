from torch import nn

class MNIST_NN(nn.Module):
    
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