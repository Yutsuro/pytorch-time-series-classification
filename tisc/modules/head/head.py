import torch.nn as nn


class BasicClassificationHead(nn.Module):
    def __init__(self,
                 num_classes: int=4,
                 num_features: int=256,
                 dropout_rate: float=0.2):
        super().__init__()
        self.head = nn.Sequential(nn.Dropout(dropout_rate),
                                  nn.Linear(num_features, 128),
                                  nn.ReLU(),
                                  nn.Dropout(dropout_rate),
                                  nn.Linear(128, 128),
                                  nn.ReLU(),
                                  nn.Dropout(dropout_rate),
                                  nn.Linear(128, num_classes))
    
    def forward(self,x):
        x = self.head(x)
        return x


class IdentityHead(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x