from src.mlp.backend import get_array_module, np

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        xp = get_array_module(x)
        self.out = 1 / (1 + xp.exp(-x))
        return self.out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
