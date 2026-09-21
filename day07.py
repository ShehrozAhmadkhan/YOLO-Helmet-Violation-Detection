import torch

"""
x1 = 0.9
x2 = 0.8
w = 0.5
b = 0.5

def sigmoid(z):
    z = torch.tensor(z)
    return 1 / (1 + torch.exp(-z))

h11 = x1*w + x2*w + b
zh11 = sigmoid(h11)
print(zh11)

h12 = x1*w + x2*w + b
zh12 = sigmoid(h12)
print(zh12)

o1 = zh11*w + zh12*w + b
z01 = sigmoid(o1)
print(z01)
"""

