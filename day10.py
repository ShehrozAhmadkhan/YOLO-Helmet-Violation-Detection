import torch

X = torch.tensor([1.0, 2.0, 1.5])
W1 = torch.tensor([[0.2, 0.1, 0.3, 0.4],
      [0.5, 0.2, 0.1, 0.3],
      [0.3, 0.4, 0.2, 0.1]],requires_grad=True)

b1 = torch.tensor([0.1, 0.2, 0.1, 0.3],requires_grad=True)

def sigmoid(z):
    return 1 / (1+torch.exp(-z))

z1 = X @ W1 + b1
HL = sigmoid(z1)
#print(f"z1: {z1}")
#print(f"HL: {HL}")

W2 = torch.tensor([[0.4, 0.2, 0.3],
      [0.1, 0.5, 0.2],
      [0.3, 0.1, 0.4],
      [0.2, 0.3, 0.1]],requires_grad=True)

b2 = torch.tensor([0.2, 0.1, 0.3], requires_grad=True)

z2 = HL @ W2 + b2
#print(f"z2: {z2}")

o1 = torch.exp(z2[0])/torch.sum(torch.exp(z2))
o2 = torch.exp(z2[1])/torch.sum(torch.exp(z2))
o3 = torch.exp(z2[2])/torch.sum(torch.exp(z2))
#print(o1)
#print(o2)
#print(o3)

loss = -torch.log(o2)
print(f"loss: {loss}")

loss.backward()
print(f"W1: {W1.grad}")
print(f"b1: {b1.grad}")
print(f"W2: {W2.grad}")
print(f"b2: {b2.grad}")




