import torch

x1 = torch.tensor(2.0)
x2 = torch.tensor(3.0)
w11 = torch.tensor(0.1,requires_grad=True)
w21 = torch.tensor(0.4,requires_grad=True)
w12 = torch.tensor(0.2,requires_grad=True)
w22 = torch.tensor(0.3,requires_grad=True)
bh1 = torch.tensor(0.1,requires_grad=True)
bh2 = torch.tensor(0.2,requires_grad=True)
wo11 = torch.tensor(0.5,requires_grad=True)
wo21 = torch.tensor(0.3,requires_grad=True)
wo12 = torch.tensor(0.2,requires_grad=True)
wo22 = torch.tensor(0.7,requires_grad=True)
bo1 = torch.tensor(0.1,requires_grad=True)
bo2 = torch.tensor(0.2,requires_grad=True)

def sigmoid(z):
    return 1/(1+torch.exp(-z))


h1z1 = x1*w11 + x2*w21 + bh1
h1 = sigmoid(h1z1)
print(f"H1: {h1}")

h1z2 = x1*w12 + x2*w22 + bh2
h2 = sigmoid(h1z2)
print(f"H2: {h2}")

olz1 = h1*wo11 + h2*wo21 + bo1

olz2 = h1*wo12 + h2*wo22 + bo2

o1 = torch.exp(olz1)/(torch.exp(olz1) + torch.exp(olz2))
print(f"O1: {o1}")

o2 = torch.exp(olz2) / (torch.exp(olz1) + torch.exp(olz2))
print(f"O2: {o2}")

loss = -torch.log(o1)
print(f"Loss: {loss}")

loss.backward()

with torch.no_grad():
    w11 -= 0.1*w11.grad
    w12 -= 0.1*w12.grad
    w21 -= 0.1*w21.grad
    w22 -= 0.1*w22.grad
    bh1 -= 0.1*bh1.grad
    bh2 -= 0.1*bh2.grad
    wo11 -= 0.1*wo11.grad
    wo12 -=0.1*wo12.grad
    wo21 -= 0.1*wo21.grad
    wo22 -= 0.1*wo22.grad
    bo1 -= 0.1*bo1.grad
    bo2 -= 0.1*bo2.grad

h1 = x1*w11 + x2*w21 + bh1
h1z = sigmoid(h1)
print(f"New H1: {h1z}")

h2 = x1*w12 + x2*w22 + bh2
h2z = sigmoid(h2)
print(f"New H2: {h2z}")

o1 = h1z*wo11 + h2z*wo21 + bo1
o2 = h1z*wo12 + h2z*wo22 + bh2

o1z = torch.exp(o1)/(torch.exp(o1)+torch.exp(o2))
o2z = torch.exp(o2)/(torch.exp(o1)+torch.exp(o2))
print(f"New O1: {o1z}")
print(f"New O2: {o2z}")
new_loss = -torch.log(o1z)
print(f"New Loss: {new_loss}")