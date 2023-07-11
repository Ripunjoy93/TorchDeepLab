import torch
import torch.nn as nn

"""
Model pipeline using pytorch functionalities
1. Design model (input, output size, forward pass)
2. Construct loss and optimizer
3. Training loop
    - forward pass: compute predictions
    - backward pass: gradients
    - update weights
"""
# f = w * x; f = 2 * x (say)

# x = torch.tensor([1,2,3,4], dtype=torch.float32)
# y = torch.tensor([2,4,6,8], dtype=torch.float32)
x = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

n_samples, n_features = x.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features

# w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


# model = nn.Linear(input_size, output_size)

# custom model
class LinearR(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearR, self).__init__()
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.lin(x)
    
model = LinearR(input_size, output_size)

x_test = torch.tensor([5], dtype=torch.float32)
print(f'prediction before training: f(5) = {model(x_test).item():.3f}') #item works when it's scalar

# training
learning_rate = 0.01
n_iters = 1000

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    # prediction= forward pass
    y_pred = model(x)
    
    # loss
    l = loss(y, y_pred)
    
    # gradients
    l.backward() # dl/dw
    
    # update weights
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()
    
    if epoch % 50 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'prediction after training: f(5) = {model(x_test).item():.3f}')