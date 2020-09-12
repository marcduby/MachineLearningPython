# imports
import torch

# Our "model"
x = torch.tensor([1., 2.], requires_grad=True)
y = 100*x

# Compute loss
loss = y.sum()

# Compute gradients of the parameters w.r.t. the loss
print("pre backward: {}, {}".format(x, x.grad))     # None
loss.backward()      
print("post backward: {}, {}".format(x, x.grad))     # tensor([100., 100.])

# MOdify the parameters by subtracting the gradient
optim = torch.optim.SGD([x], lr=0.001)
print(x)        # tensor([1., 2.], requires_grad=True)
optim.step()
print(x)        # tensor([0.9000, 1.9000], requires_grad=True)
