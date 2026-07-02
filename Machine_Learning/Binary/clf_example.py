import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 각 샘플은 4개의 특성을 가지고 있으며, 8개의 샘플이 존재
# 8x4
x_train = [
    [1, 2, 1, 1],
    [2, 1, 3, 1],
    [3, 1, 3, 4],
    [4, 1, 5, 6],
    [1, 7, 8, 5],
    [1, 3, 5, 6],
    [1, 6, 6, 6],
    [1, 7, 7, 7]
]
# output: 8x3
y_train = [2, 2, 2, 1, 1, 1, 0, 0] # one-hot encoding으로 변경
# unsqueeze(0) => 1x8
# unsqueeze(1) => 8x1    ## 0은 항상 row 방향, 1은 항상 column 방향 !

"""
[[0, 0, 1],
[0, 0, 1],
[0, 0, 1],
[0, 1, 0], .... [1, 1, 0]]
"""
x_train = torch.FloatTensor(x_train)
# one-hot encoding: F.onehot, pd.dummies, 직접 가공..
# 내가 코드를 만들면 F.onehot 사용..
# github에 프로젝틀르 참고를 많이하게 됨.
# y_onehot = torch.zeros(8, 3)
# y_onehot = y_onehot.scatter(1, y_train.unsqueeze(1), 1)
y_train = torch.LongTensor(y_train)
y_onehot= F.one_hot(y_train, 3)
# print(x_train.shape)
# print(y_onehot)

# 모델 파라미터 초기화 (w, b)
# softmax(wx+b) ==> softmax(x_train.matmul(w) + b)
w = torch.zeros((4, 3), requires_grad=True)
# 1: broadcast 방식으로 다 더해짐. 3: 각각의 bias 존재
b = torch.zeros(3, requires_grad=True)
# SGD(특정한 목적이 아니면 학습용도로 많이 사용 < "Adam": 무지성으로 그냥 많이 지정해서 사용
optimizer = optim.SGD([w, b], lr=0.01)

nb_epoch = 1000
for epoch in range(nb_epoch+1):
    hx = F.softmax(x_train.matmul(w) + b, dim=1)
    loss = (y_onehot * -torch.log(hx)).mean()
    optimizer.zero_grad()
    loss.backward()  # backwarding
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}: Loss {loss.item():.4f}')

# for param in [w, b]:
#     print(param)

