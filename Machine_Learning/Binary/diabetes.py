import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

data_diabetes = pd.read_csv('data-03-diabetes.csv', header=None)

# 입력값: x / 정답: y
x_diabetes = data_diabetes.iloc[:, :-1].values
y_diabetes = data_diabetes.iloc[:, -1].values

# 마지막 row 20개 => 평가용(test)로 분리 / 20개 제외한 나머지=> 학습용(train)
x_train_db = x_diabetes[:-20]
y_train_db = y_diabetes[:-20]

x_test_db = x_diabetes[-20:]
y_test_db = y_diabetes[-20:]

# 텐서(Tensor)로 변환
x_train_tensor = torch.FloatTensor(x_train_db)
y_train_tensor = torch.FloatTensor(y_train_db).view(-1, 1)

x_test_tensor = torch.FloatTensor(x_test_db)


from Own_ML_class import BinaryClassification
in_dim = x_train_db.shape[1]
model = BinaryClassification(in_dim=in_dim, out_dim=1)


loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 모델 학습
epochs = 2000
for epoch in range(epochs):

    hypothesis = model(x_train_tensor)

    # 정답과 예측값이 얼마나 다른지 평균 제곱 오차로 계산
    cost = loss_fn(hypothesis, y_train_tensor)

    optimizer.zero_grad()
    cost.backward()       # backward: 오차를 바탕으로 미분
    optimizer.step()

# 마지막 20개 행 평가하기
predict_prob = model(x_test_tensor)
correct_count = 0

# for문 사용 =>  20개 데이터 비교
for i in range(20):
    # 예측 확률값 0.5 이상이면 1(당뇨) / 아니면 0(정상)
    if predict_prob[i].item() > 0.5:
        prediction = 1
    else:
        prediction = 0
    # 정답(y_test_cb[i])과 예측값이 같은지 확인
    if prediction == y_test_db[i]:
        correct_count = correct_count + 1 # 정답 맞추면 +1

# 정확도 계산
accuracy = (correct_count / 20) * 100

print(f"당뇨병 모델 맞춘 개수: {int(correct_count)} / 20")
print(f"당뇨병 모델 최종 정확도: {accuracy}%")