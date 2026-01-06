import numpy as np
from src.mlp.layers import Dense
from src.mlp.activations import ReLU, Sigmoid
from src.mlp.losses import SoftmaxWithLoss
from src.mlp.backend import to_cpu, get_array_module

class MLP:
    def __init__(self, input_size=784, hidden_size_list=[100], output_size=10, use_activation='fast_relu'):
        self.params = {}
        self.layers = []
        
        # 層の構築
        dims = [input_size] + hidden_size_list + [output_size]
        
        for i in range(len(dims)-1):
            self.layers.append(Dense(dims[i], dims[i+1]))
            if i < len(dims) - 2: # 最後の層の後には活性化関数を入れない (SoftmaxWithLoss が扱うため)
                if use_activation == 'sigmoid':
                    self.layers.append(Sigmoid())
                elif use_activation == 'relu':
                    self.layers.append(ReLU())
        
        self.last_layer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
        
    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)
    
    def accuracy(self, x, t):
        xp = get_array_module(x)
        y = self.predict(x)
        y = xp.argmax(y, axis=1)
        if t.ndim != 1 : t = xp.argmax(t, axis=1)
        
        accuracy = xp.sum(y == t) / float(x.shape[0])
        # 結果はスカラーだが、cupy array の可能性があるので to_cpu するか float() でキャスト
        return float(accuracy)
        
    def gradient(self, x, t):
        # 順伝播 (Forward)
        self.loss(x, t)
        
        # 逆伝播 (Backward)
        dout = 1
        dout = self.last_layer.backward(dout)
        
        layers = reversed(self.layers)
        for layer in layers:
            dout = layer.backward(dout)
            
        # 勾配の収集
        all_params = []
        all_grads = []
        
        for layer in self.layers:
            if isinstance(layer, Dense):
                all_params.append(layer.W)
                all_params.append(layer.b)
                all_grads.append(layer.dW)
                all_grads.append(layer.db)
                
        return all_params, all_grads
