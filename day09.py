import torch
X = torch.tensor([[2.0, 3.0]]) 
W1 = torch.tensor([[0.1, 0.2],
                    [0.4, 0.3]], requires_grad=True)  
b1 = torch.tensor([0.1, 0.2], requires_grad=True) 

def sigmoid(z):
    return 1/(1+torch.exp(-z))

H = sigmoid(X @ W1 + b1)
print(H)
W2 = torch.tensor([[0.5, 0.2],
                    [0.3, 0.7]], requires_grad=True)   # shape: (2, 2)
b2 = torch.tensor([0.1, 0.2], requires_grad=True)   # bo1=0.1, bo2=0.2
O = H @ W2 + b2
O1 = torch.exp(O[0][0])/torch.sum(torch.exp(O))
O2 = torch.exp(O[0][1])/torch.sum(torch.exp(O))
print(O1)
print(O2)

loss = -torch.log(O1)
print(loss)

loss.backward()
print(f"w11: {W1.grad}")
print(f"w12: {W2.grad}")