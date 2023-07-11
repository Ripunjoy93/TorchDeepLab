import torch
import numpy as np

# Check whether GPU available
print(f"Is cuda available: {torch.cuda.is_available()}")

# Empty Tensors
x = torch.empty(1) #scalar
print(f"Scalar tensor: {x}")

x = torch.empty(3) #1-D vector
print(f"1-D vector: {x}")

x = torch.empty(3,4) #2-D matrix
print(f"2-D matrix: {x}")
# and so on

# Random tensors
x = torch.rand(2,2)
print(f"2-D matrix random: {x}")

# Other tensors
x = torch.ones(3,3, dtype=torch.int)
print(f"2-D matrix with 1s: {x}")

# check size
print(f"Check size: {x.size()}")

# Tensor from python list
x = torch.tensor([1.2, 2.5, 3.0])

# Addition
x = torch.rand(2,2)
y = torch.rand(2,2)

z = x + y #element wise addition
x = torch.add(x,y)
y.add_(x) #inplace addition

# every function with trailing underscore does inplace operation

# Other operations
# subtraction: -, sub
# multiplication: *, mult
# division: /, div

# Slicing
x = torch.rand(5,3)
print(f"Slicing: {x[:,0]}")

# Reshaping
x = torch.rand(4,4)
y = x.view(16)
print(f"Reshaping: {y}")
y = x.view(-1,8) #torch automatically defines the other dimension as 2
print(f"Reshaping: {y}")

# Convert to numpy array
y = x.view(16)
print(f"To numpy: {y.numpy()}")
# CAREFUL: if tensor is in CPU then tensor and the numpy array will refer to the same memory location, modifying 1 will modify the other

# From numpy to tensor
b = torch.from_numpy(np.ones(5))

# How to create tensor in GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device, dtype=torch.double)
    # OR move to device
    y = torch.ones(5)
    y = y.to(device)
    z = x + y
    # now to convert to numpy we need to move z to cpu
    z = z.to("cpu")
    z1 = z.numpy()
    
