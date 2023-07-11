import torch

# How to calculate gradients in pytorch using autograd package
x = torch.rand(3, requires_grad=True) #by specifying it as true we let torch know that gradient is needed for some functions
print(f"Grad required: {x}") #creates a computational graph, with back-propagation we can calculate the gradients

y = x + 2
print(y)

z = y * y * 2
z = z.mean() # scalar, when it's a vector we need a dummy vector of same size which needs to be passed in the backward function
print(z)

z.backward() # dz/dx
print(x.grad)

# Prevent gradient tracking when not needed in our computational graph
# x.requires_grad_(False)
# x.detach()
# with torch.no_grad():

# Training example: gradients being accumulated we need to empty the grads before next iterations

weights = torch.ones(3, requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)


weights = torch.ones(3, requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_()
