import torch

#actual = torch.tensor([10.0, 20.0, 30.0])
#predicted = torch.tensor([7.0, 25.0, 28.0])
"""
mse = (torch.sum((actual - predicted)**2))/len(actual)
print(mse)

mse = torch.mean((actual - predicted)**2)
print(mse)
"""

#mse = torch.mean((actual - predicted)**2)
#print(mse)

a = torch.tensor(1.0)
p = torch.tensor(0.75)

cel = -(a*torch.log(p) + (1-a)*(torch.log(1-p)))
print(cel)

predicted_probabilities = torch.tensor([0.05, 0.10, 0.15, 0.60, 0.10])
actual_class_index = 3   # dataset se mila label — sahi class index 3 hai
output = -torch.log(predicted_probabilities[3])
print(output)