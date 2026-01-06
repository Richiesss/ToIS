from src.mlp.backend import get_array_module, np

def softmax(x):
    xp = get_array_module(x)
    if x.ndim == 2:
        x = x.T
        x = x - xp.max(x, axis=0)
        y = xp.exp(x) / xp.sum(xp.exp(x), axis=0)
        return y.T
    
    x = x - xp.max(x)
    return xp.exp(x) / xp.sum(xp.exp(x))

def cross_entropy_error(y, t):
    xp = get_array_module(y)
    
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # t が one-hot encoding の場合、インデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    # cupy の場合、fancy indexing は numpy と互換性があるが注意が必要
    # arange の作成
    return -xp.sum(xp.log(y[xp.arange(batch_size), t] + 1e-7)) / batch_size

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmax の出力
        self.t = None # 正解ラベル (one-hot または index)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        xp = get_array_module(self.y)
        
        if self.t.size == self.y.size: # one-hot ベクトルの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[xp.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx
