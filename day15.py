import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3,4)
        self.layer2 = nn.Linear(4,3)

    def forward(self,x):
        z1 = self.layer1(x)
        a1 = torch.sigmoid(z1)
        z2 = self.layer2(a1)
        return z2

torch.manual_seed(42)
NN = NeuralNetwork()
x = torch.rand(600,3)
#y = torch.randint(0,3,(600,))
y = torch.argmax(x, dim=1)
dataset = TensorDataset(x,y)
loader = DataLoader(dataset,batch_size=50,shuffle=True)
optimizer = torch.optim.Adam(NN.parameters(),lr = 0.01)
loss_fn = nn.CrossEntropyLoss()

for i in range(100):
    for xb,yb in loader:
        z2 = NN(xb)
        loss = loss_fn(z2,yb)
        #if i%10==0:
            #print(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

"""       
for xb,yb in loader:
    break

with torch.no_grad():
    z2 = NN(xb)
    pred = torch.argmax(z2,dim=1)
    print(pred[:10])
    print(yb[:10])
    correct = (pred == yb).sum()
    accuracy = correct/len(yb)
    print(accuracy)

"""
count = 0
total_count = 0

with torch.no_grad():
    for xb,yb in loader:
        z2 = NN(xb)
        pred = torch.argmax(z2,dim=1)
        correct = (pred == yb).sum()
        count += correct
        total_count += len(xb)

accuracy = count / total_count
print(accuracy)
