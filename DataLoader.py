import pandas as pd
import torch


class DataLoader:
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, miniBatchSize=-1):
        self.X = X
        self.Y = Y
        self.y_col = ""
        self.miniBatchSize = X.shape[0] - 1 if miniBatchSize == -1 else miniBatchSize
        self.n = X.shape[0]
        assert self.n == Y.shape[0]
        self.pos = 0

    def get(self):
        if self.n <= self.pos + self.miniBatchSize:
            self.pos = 0
            return None, None
        else:
            x, y = (self.X[self.pos:self.pos + self.miniBatchSize],
                    self.Y[self.pos:self.pos + self.miniBatchSize])
            self.pos += self.miniBatchSize
            return x, y
