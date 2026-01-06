import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.data_loader import load_mnist

def check_data_distribution():
    print("データの分布を確認中...")
    (X_train, y_train), (X_test, y_test) = load_mnist()
    
    print(f"学習データ: {X_train.shape}")
    print(f"学習ラベルの先頭20個: {y_train[:20]}")
    
    # Check if labels are sorted
    if np.all(y_train[:-1] <= y_train[1:]):
        print("警告: 学習データがソートされている可能性があります！シャッフルが機能していないかもしれません。")
    else:
        print("学習データはシャッフルされているようです。")
        
    print(f"テストデータ: {X_test.shape}")
    print(f"テストラベルの先頭20個: {y_test[:20]}")
    
    # Check label balance
    unique, counts = np.unique(y_train, return_counts=True)
    print("学習データのラベル分布:")
    for u, c in zip(unique, counts):
        print(f"  {u}: {c}")

if __name__ == "__main__":
    check_data_distribution()
