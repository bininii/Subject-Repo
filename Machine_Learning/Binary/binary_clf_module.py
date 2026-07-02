import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

# x_data: 6x2
# y_data: 6x1
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

"""
nn.sequential(): nn.Module 층을 차례로 쌓을 수 있도록 하는 것
keras (tensorflow) 유사하게 사용, 딥러닝 층을 여러 층을 쌓기때문에 sequential 내에 layer 층 쌓음
쉽게 여러 함수들을 연결 시켜주는 기능

"""
# 모델 생성
model = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
)
# model.parameters(): ?? 2x1: w, 1: b
optimizer = optim.SGD(model.parameters(), lr=1)
for params in model.parameters():
    print(params) # 초기 모델 파라미터 값 확인

    nb_epoch = 3000
    for epoch in range(nb_epoch + 1):
        # pass
        # 1. hypothesis (sigmoid)
        hx = model(x_train) # sigmoid(wx+b)) 계산

        # 2. cost function (-y(log(h(x)) - (1-y)log(1-h(x))) (기계학습 전달 ppt-page.15 슬라이드 참고)
        # loss = -(y_train * torch.log(hx) + (1 - y_train) * torch.log(1 - hx)).mean()
        loss = F.binary_cross_entropy(hx, y_train) # loss function 을 통해 error 값 계산

        # 3. gradient descent algorithm & update param
        optimizer.zero_grad()
        loss.backward()  # loss 함수 미분 (w, b 각각에 대해서 편미분) -(y_train*torch.log(hx)+(1-y_train)*torch.log(1-hx)).mean() 룰 미분적용
        optimizer.step() # w = w - alpha(w로 미분한 값), b = b - alpha(b로 미분한 값)

        # 4. cost 값이 잘 떨어지고 있는지 확인
        if epoch % 100 == 0:
            print(f'epoch: {epoch}, loss: {loss}')




