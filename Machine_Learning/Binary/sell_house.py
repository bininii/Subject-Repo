##### 수정필요

import pandas as pd
import numpy as np
import torch
from torch import nn, optim

# sell_house.txt 파일 불러오기
data = np.loadtxt(r"D:\bin ai-engr pycharm\binary\sell_house.txt")

x = data[:, 1 : -1]
y = data[:, -1]

x_train, x_test = x[:-5], x[-5:]
y_train, y_test = y[:-5], y[-5:]

# 세로 기둥 형태로 바꾸기
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# 공식 대입 (w = (X^T * X)^(-1) * X^T * y)
temp1 = np.matmul(x_train.T, x_train)
temp2 = np.linalg.inv(temp1)
temp3 = np.matmul(temp2, x_train.T)
w = np.matmul(temp3, y_train)

# 문제 예측 결과
prediction = np.matmul(x_test, w)

# 무작위 값이 매번 똑같이 나오도록 고정
torch.manual_seed(42)

# 데이터의 평균, 표준편차를 이용하여 크기 맞추기
x_mean = x_train.mean(axis=0)
x_std = x_train.std(axis=0)

x_train_scaled = (x_train - x_mean) / x_std
x_test_scaled = (x_test - x_mean) / x_std

# 텐서(Tensor)로 변환
x_train_t = torch.FloatTensor(x_train_scaled)
y_train_t = torch.FloatTensor(y_train)
x_test_t = torch.FloatTensor(x_test_scaled)


# 선형 회귀(Linear) 모델과 오차 계산기, ############공부 방법(SGD) 설정###########
model = nn.Linear(11, 1)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 5000번 반복 학습 시작
epochs = 5000
for epoch in range(epochs + 1):
    pred = model(x_train_t)          # 예측해보고
    loss = loss_fn(pred, y_train_t)  # 오차가 얼마나 되는지 계산

    optimizer.zero_grad()            # 지난 공부 기억 리셋
    loss.backward()                  # 오차를 바탕으로 반성하기
    optimizer.step()                 # 반성한 내용을 바탕으로 모델 수정

    # 1000번마다 한 번씩 현재 오차(loss) 출력하기
    if epoch % 1000 == 0:
        print(f"epoch: {epoch} loss: {loss.item()}")

# 예측 결과
ml_pred = model(x_test_t).detach().numpy()

print("\nML 예측:")
print(ml_pred)
print("-" * 30)

# 소수점 둘째 자리(:.2f)까지 출력
print("GT\t\tML\t\tProjection")
for i in range(len(y_test)):
    gt = y_test[i][0]
    ml = ml_pred[i][0]
    proj = prediction[i][0]
    print(f"{gt:.1f}\t{ml: .2f}\t\t{proj: .2f}")

# 평균 제곱 오차(MSE) 계산 및 출력
ml_mse = np.mean((ml_pred - y_test) ** 2)
proj_mse = np.mean((prediction - y_test) ** 2)

print(f"ML MSE: {ml_mse}")
print(f"Projection MSE: {proj_mse}")