import torch
import numpy as np

x1 = 1
x2 = 4
w1 = 0.4
w2 = 0.2
bias = 0.1

f = (w1*x1) + (w2*x2) 

z = f + bias

def sigmoid(z):
    output = 1 / (1 + np.exp(z))
    return output

def tanh(z):
    output = (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))
    return output

def relu(z):
    output = max(0,z)
    return output

"""
print(f"s: {sigmoid(z)}")
print(f"tanh: {tanh(z)}")
print()
print(np.tanh(z))
print(f"relu: {relu(z)}")
"""

z = torch.tensor([0.70,0.60,0.35,0.50,0.70,0.30,0.60,0.10,0.75,0.90])
#print(z)
exp_z = torch.exp(z)

sum_exp = torch.sum(exp_z)
#print(sum_exp)

softmax = exp_z/ sum_exp
print(softmax)
print(torch.sum(softmax))