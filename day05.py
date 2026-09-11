import torch
import numpy as np

x1 = 1
x2 = 4
w1 = 0.4
w2 = 0.2
bias = 0.1

y = x1*w1 + x2*w2
z = y + bias

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def relu(z):
    y = max(0,z)
    return y

print(f"sigmoid: {sigmoid(z)} relu: {relu(-3)}")

def tanh(z):
    y = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    return y

print(f"tanh function: {tanh(z)}  tanh built in: {np.tanh(z)}")