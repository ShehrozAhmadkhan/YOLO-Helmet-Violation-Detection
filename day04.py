import numpy as np

x1 = 5
x2 = 6

w1 = 0.4
w2 = 0.2

bias = 0.1

step1 = (x1*w1) + (x2*w2)

step2 = step1 + bias

def activation_function(value):
    if value >= 0:
        return 1
    else:
        return 0

step3 = activation_function(step2)
print(f"Output: {step3}")