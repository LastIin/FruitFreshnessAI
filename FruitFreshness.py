import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name)

#scalar
scalar = torch.tensor(7)
print(scalar.ndim)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

ran_tensor = torch.rand((3,4), device = device)

def tensor_detail(t):
    print(t)
    print(f"Datatype of tensor: {t.dtype}")
    print(f"Shape of tensor {t.shape}")
    print(f"Device tensor is on: {t.device}")

tensor_detail(ran_tensor)