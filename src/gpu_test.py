import sys
import os

try:
    import cupy as cp
    import numpy as np
    
    print("CuPy check...")
    x_gpu = cp.array([1, 2, 3])
    y_gpu = cp.array([4, 5, 6])
    z_gpu = x_gpu + y_gpu
    
    print(f"GPU Array Result: {z_gpu}")
    print(f"Device: {cp.cuda.runtime.getDeviceCount()} devices found.")
    print("Success: CuPy is working and can access GPU.")
    
except ImportError as e:
    print(f"ImportError: {e}")
    print("CuPy is not installed or libraries are missing.")
except Exception as e:
    print(f"Error: {e}")
    # Print traceback
    import traceback
    traceback.print_exc()

