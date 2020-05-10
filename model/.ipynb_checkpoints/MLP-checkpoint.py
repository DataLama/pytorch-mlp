import torch.nn as nn
import torch.nn.functional as F

class Perceptron(nn.Module):
    """Multi-Layer Perceptron 2-hidden layer"""
    def __init__(self, vocab_size, label):
        super(Perceptron, self).__init__()
        self.linear1 = nn.Linear(vocab_size, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, label)

    def forward(self, X):
        y_pred = self.linear1(X)
        y_pred = F.relu(y_pred)
        y_pred = self.linear2(y_pred)
        y_pred = F.relu(y_pred)
        y_pred = self.linear3(y_pred)
        return y_pred