import torch

# Chain Rule: dz/dx = dz/dy * dy/dx
# Computational Graph: compute local gradients
# Backward pass: compute dLoss/dWeights using chain rule

x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad=True)

# forward pass and compute the loss
y_hat = w * x
loss = (y_hat - y) ** 2

print(loss)

# backward pass
loss.backward()
print(w.grad)

# update the weights
# then next forward and backward pass