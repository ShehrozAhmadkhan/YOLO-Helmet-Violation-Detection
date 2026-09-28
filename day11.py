import torch
import torch.nn as nn

"""
layer = nn.Linear(3,4)
print(layer.weight)
print(layer.weight.shape)
print(layer.bias)
print(layer.bias.shape)
"""

X = torch.tensor([[1.0,2.0,1.5]])
layer1 = nn.Linear(3,4)
h1 = layer1(X)
#print(f"h1: {h1}")
#print(h1.shape)
z1 = torch.sigmoid(h1)
#print(f"z1 sigmoid: {z1}")

#print()
layer2 = nn.Linear(4,3)
o2 = layer2(z1)
#print(f"o2: {o2}")


z2 = torch.softmax(o2,dim=1)
#print(f"z2: {z2}")
#print(torch.sum(z2))

print(-torch.log(z2))
print()
loss = -torch.log(z2[0][1])
print(loss)