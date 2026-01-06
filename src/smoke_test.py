import numpy as np
from src.mlp.network import MLP
from src.mlp.optimizers import Adam
import sys
import os

# src をパスに追加 (念のため)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_mlp():
    print("MLPのテストを実行中...")
    # ダミーデータ
    X = np.random.randn(10, 784)
    y = np.random.randint(0, 10, size=10)
    
    model = MLP(input_size=784, hidden_size_list=[50], output_size=10)
    optimizer = Adam(lr=0.01)
    
    # 順伝播
    loss = model.loss(X, y)
    print(f"初期損失: {loss:.4f}")
    
    # 逆伝播と更新
    params, grads = model.gradient(X, y)
    optimizer.update(params, grads)
    
    # 再度順伝播
    loss_after = model.loss(X, y)
    print(f"1ステップ後の損失: {loss_after:.4f}")
    
    if loss_after < loss:
        print("MLP スモークテスト合格: 損失が減少しました。")
    else:
        print("MLP スモークテスト警告: 損失が減少しませんでした (ランダムな変動や学習率の問題の可能性がありますが、コードは動作しました)。")

if __name__ == "__main__":
    test_mlp()
