import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

"""
torch.manual_seed(42)
x = torch.rand(600,3)
y = torch.randint(0,3,(600,))
dataset = TensorDataset(x,y)
print(len(dataset))
loader = DataLoader(dataset,batch_size=50,shuffle=True)
print(len(loader))

for xb1,yb1 in loader:
    print(xb1.shape)
    print(yb1.shape)
    break
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

x = torch.rand(600,3)
y = torch.randint(0,3,(600,))

dataset = TensorDataset(x,y)
loader = DataLoader(dataset,batch_size=50,shuffle=True)
optimizer = torch.optim.Adam(NN.parameters(), lr = 0.01)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(3):
    for xb,yb in loader:
        z2 = NN(xb)
        loss = loss_fn(z2,yb)
        print(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
 

"""
for xb,yb1 in loader:
    z2 = NN(xb)
    a2 = torch.softmax(z2,dim=1)
    #label = yb1[0]
    #print(label)
    #print(a2[0])
    #print(a2[0][label])
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(z2,yb1)
    print(loss)
    break
    
"""

   