import numpy as np
import torch
import torch.nn.functional as F
import time
from core.operations import WinogradConv
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--thrs', type=float, default=1e-2, help='Threshold for the test case')

winograd = WinogradConv()
def test_winograd_conv_layer(thrs=1e-2):
    X = np.random.rand(1, 3, 224, 224)     
    W = np.random.rand(4, 3, 3, 3)     

    X_tensor = torch.from_numpy(X).float()
    W_tensor = torch.from_numpy(W).float()

    
    start_time1 = time.time()
    Y_pytorch = F.conv2d(X_tensor, W_tensor, stride=1)
    end_time1 = time.time()

    
    start_time = time.time()
    Y = winograd.convolve(X, W,1)
    end_time = time.time()

    
    check1 = np.allclose(Y, Y_pytorch.detach().numpy(), atol=thrs)
    check2 = Y.shape == Y_pytorch.shape

    if not check1:
        print("test case failed: Y and Y_pytorch are not close")
    if not check2:
        print("test case failed: Y and Y_pytorch are not the same shape")
    else:
        print("test case passed")
        print("PyTorch time:", end_time1 - start_time1)
        print("Winograd time:", end_time - start_time)

if __name__ == "__main__":
    
    args = parser.parse_args()
    

    
    test_winograd_conv_layer(thrs=args.thrs)
