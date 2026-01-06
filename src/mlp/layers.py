import numpy as np
from src.mlp.backend import get_array_module, to_gpu, to_cpu

class Dense:
    def __init__(self, input_size, output_size, weight_init_std=0.01):
        # 初期化は NumPy で行う
        self.W = weight_init_std * np.random.randn(input_size, output_size)
        self.b = np.zeros(output_size)
        
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        # 入力データのモジュール (numpy or cupy) に合わせる
        xp = get_array_module(x)
        
        # もし入力が GPU で、重みがまだ CPU なら GPU に送る (簡易的な自動転送)
        if xp != get_array_module(self.W):
            if xp.__name__ == 'cupy':
                self.W = to_gpu(self.W)
                self.b = to_gpu(self.b)
            else:
                self.W = to_cpu(self.W)
                self.b = to_cpu(self.b)

        self.x = x
        out = xp.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        xp = get_array_module(dout)
        dx = xp.dot(dout, self.W.T)
        self.dW = xp.dot(self.x.T, dout)
        self.db = xp.sum(dout, axis=0)
        return dx
