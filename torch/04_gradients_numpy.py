import numpy as np

# Manually using numpy (numerical)
# f = w * x; f = 2 * x (say)

x = np.array([1,2,3,4], dtype=np.float32)
y = np.array([2,4,6,8], dtype=np.float32)

w = 0.0

# model prediction
def forward(x):
    return w * x

# loss = MSE
def loss(y, y_predicted):
    return ((y_predicted-y) ** 2).mean()

# gradient
# MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N 2x (w*x - y)
def gradient(x,y,y_predicted):
    return np.dot(2*x, y_predicted-y).mean()

print(f'prediction before training: f(5) = {forward(5):.3f}')

# training
learning_rate = 0.01
n_iters = 30

for epoch in range(n_iters):
    # prediction= forward pass
    y_pred = forward(x)
    
    # loss
    l = loss(y, y_pred)
    
    # gradients
    dw = gradient(x,y,y_pred)
    
    # update weights
    w -= learning_rate * dw
    
    if epoch % 3 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'prediction after training: f(5) = {forward(5):.3f}')