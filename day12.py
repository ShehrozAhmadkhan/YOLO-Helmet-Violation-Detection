import torch
import torch.nn as nn
"""
class Student:
    def __init__(self, name, marks):
        self.name = name
        self.marks = marks

    def result(self):
        if self.marks >= 50:
            return self.name + " pass"
        else:
            return self.name + " fail"

s1 = Student("Ali", 80)
s2 = Student("Sara", 45)

print(Student.result(s1))
print(s1.result())
"""

"""
class Person:
    def __init__(self, name):
        self.name = name

    def hello(self):
        return "Hello " + self.name

class Student(Person):
    def __init__(self, name, marks):
        super().__init__(name)
        self.marks = marks

s1 = Student("Ali", 80)
print(s1.hello())
print(s1.marks)
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


l1 = MyNetwork()
#print(l1.layer1)
#print(l1.layer2)
X = torch.tensor([[1.0, 2.0, 1.5]])
f = l1(X)
probs = torch.softmax(f,dim=1)
#print(probs)
loss = -torch.log(probs[0][1])
#print(loss)
loss.backward()
print(l1.layer1.weight.grad)
print(l1.layer2.weight.grad)
#print(probs.sum())
#print(f)
#print(f.shape)
#print(f.sum())

#for p in l1.parameters():
#    print(p.shape)