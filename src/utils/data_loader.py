import os
import numpy as np
from sklearn.datasets import fetch_openml

def download_mnist(data_dir='data', filename='mnist.npz'):
    """
    OpenML から MNIST データセットをダウンロードし、npz 形式で保存します。
    ファイルが既に存在する場合は何もしません。
    """
    file_path = os.path.join(data_dir, filename)
    if os.path.exists(file_path):
        print(f"MNIST データセットが見つかりました: {file_path}")
        return

    print("MNIST データセットをダウンロード中... (これには時間がかかる場合があります)")
    # fetch_openml を使用して MNIST データを取得
    # version 1 は 784 次元 (28x28 フラット化済み)
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(np.int32)
    
    # ピクセル値を 0-1 に正規化
    X /= 255.0
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    np.savez_compressed(file_path, X=X, y=y)
    print(f"MNIST データセットを保存しました: {file_path}")

def load_mnist(data_dir='data', filename='mnist.npz'):
    """
    ローカルの npz ファイルから MNIST データセットを読み込みます。
    戻り値: (X_train, y_train), (X_test, y_test)
    """
    file_path = os.path.join(data_dir, filename)
    
    if not os.path.exists(file_path):
        download_mnist(data_dir, filename)
        
    data = np.load(file_path)
    X = data['X']
    y = data['y']
    
    # シャッフルと分割
    # 標準的な分割は 60k 学習用, 10k テスト用
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    X = X[indices]
    y = y[indices]
    
    train_size = 60000
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return (X_train, y_train), (X_test, y_test)

if __name__ == "__main__":
    # ダウンロードと読み込みのテスト
    download_mnist()
    (X_tr, y_tr), (X_te, y_te) = load_mnist()
    print(f"学習用データ形状: {X_tr.shape}, テスト用データ形状: {X_te.shape}")
