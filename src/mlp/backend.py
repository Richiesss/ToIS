import numpy as np

try:
    import cupy as cp
    gpu_available = True
except ImportError:
    cp = None
    gpu_available = False

def get_array_module(x):
    """
    データ x に応じて numpy または cupy モジュールを返します。
    """
    if gpu_available and isinstance(x, cp.ndarray):
        return cp
    return np

def use_gpu(use: bool):
    """
    グローバルでGPUを使用するかどうかを設定できます（今回は簡易実装のため明示的には使用しないかも）
    基本的にはデータがGPUにあるかどうかで判断します。
    """
    pass

def to_gpu(x):
    """
    データをGPUに転送します。
    """
    if not gpu_available:
        return x
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return cp.asarray(x)
    return x

def to_cpu(x):
    """
    データをCPUに転送します。
    """
    if x is None:
        return None
    if gpu_available and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return x

# 短縮エイリアス (デフォルトは numpy だが、クラス内で self.xp として使い分ける推奨)
# ただし、今回はシンプルに xp をインポートして使う形にするか、
# 各クラスで get_array_module を呼ぶ形にするか。
# Scratch実装としては、データがGPUにあれば計算もGPUで、というCuPyの流儀に従うのが楽。

def to_numpy(x):
     return to_cpu(x)
