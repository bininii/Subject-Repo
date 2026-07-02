### 2.
import numpy as np

def sigmoid(val):
    lr = 0.5
    return 1 / (1 + np.exp(-val))

lr = 0.5

x1, x2 = 0.1, 0.2
w1, w2, w3, w4, w5, w6, w7, w8 = 0.2, 0.23, 0.38, 0.35, 0.42, 0.32, 0.64, 0.6
target_o1, target_o2 = 0.4, 0.6


h1 = x1 * w1 + x2 * w3
h2 = x1 * w2 + x2 * w4

z3 = w5 * h1 + w6 * h2
dz3_w5 = h1
dz3_w6 = h2
z4 = w7 * h1 + w8 * h2
dz4_w7 = h1
dz4_w8 = h2

o1 = sigmoid(z3)
o2 = sigmoid(z4)

Eo1 = 0.5*(target_o1 - o1)**2
Eo2 = 0.5*(target_o2 - o2)**2
Etotal = Eo1 + Eo2

def backpropagation_1(weight, d_weight, target, output):
    grad = -(target - output) * (output * (1 - output)) * d_weight
    weights = weight - (lr * grad)
    return weights

if __name__ == '__main__':
    print(backpropagation_1(w5, dz3_w5, target_o1, o1))
    print(backpropagation_1(w6, dz3_w6, target_o1, o1))
    print(backpropagation_1(w7, dz4_w7, target_o2, o2))
    print(backpropagation_1(w8, dz4_w8, target_o2, o2))
