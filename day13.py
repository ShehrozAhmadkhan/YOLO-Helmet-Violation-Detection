import torch.nn as nn
import torch


class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3,4)
        self.layer2 = nn.Linear(4,3)

    def forward(self,X):
        z1 = self.layer1(X) #x@w + b
        a1 = torch.sigmoid(z1)
        z2 = self.layer2(a1)
        return z2

torch.manual_seed(42)
NN = MyNetwork()
X = torch.tensor([[1.0, 2.0, 1.5]])
optimizer = torch.optim.Adam(NN.parameters(),lr=0.01)

for i in range(50):
    a2 = torch.softmax(NN(X),dim=1)
    loss = -torch.log(a2[0][1])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if i%10 == 0:
        print(f"i: {i}, loss: {loss}")
#a2 = torch.softmax(NN(X),dim=1)
#print(a2)
#loss = -torch.log(a2[0][1])
#print(loss)
#loss.backward()
#print(NN.layer1.weight)
#print(NN.layer1.bias)
#print(NN.layer1.weight.grad)
#print(NN.layer1.bias.grad)

#for p in NN.parameters():
#    print(p.shape)
#print(NN.layer1.bias)
#print(NN.layer1.bias.grad)
#optimizer = torch.optim.SGD(NN.parameters(),lr=0.1)
#optimizer.step()
#print(NN.layer1.bias)

#print(torch.rand(2))
#torch.manual_seed(42)
#print(torch.rand(2))




"""
class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3,4)
        self.layer2 = nn.Linear(4,3)

    def forward(self,X):
        z1 = self.layer1(X)
        a1 = torch.sigmoid(z1)
        z2 = self.layer2(a1)
        return z2

torch.manual_seed(42)
NN = MyNetwork()
z2 = MyNetwork.forward(X)
a2 = torch.softmax(z2,dim=1)
actual = a2[0][1]
loss = -torch.log(actual)

loss.backward()

optimizer = torch.optim.SGD(NN.parameters(),lr=0.1)
optimizer.step()
"""