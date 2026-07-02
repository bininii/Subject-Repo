import torch
import torch. nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Dataset, Dataloader 사용
from torch.utils.data import Dataset, DataLoader, TensorDataset

# 0차원 (스칼라, Scalar): 숫자 하나 ($5$)1차원
# (벡터, Vector): 숫자를 한 줄로 나열한 것 ($[1, 2, 3]$) => 2차원
# (행렬, Matrix): 숫자를 가로세로 표 형태로 나열한 것 => 3차원 이상
# (텐서, Tensor): 행렬을 여러 개 겹쳐 놓은 입체적인 구조

# simple data
# 3번의 평가 점수를 통한 최종 성적 예측 - 3x4 행렬
# example: 3번의 평가 점수와 공부시간
# [x1, x2, x3...]
# features 개수가 몇개가 늘어나든지 상관없이 matmul 생성
x_train = torch.FloatTensor([
    [73, 80, 75, 3],
    [90, 85, 95, 5],
    [89, 90, 95, 10],
    [96, 100, 100, 1]
])

# 최종점수
# 출력 2개여도 상관없음
y_train = torch.FloatTensor([
    [152],
    [185],
    [186],
    [198]
])

dataset = TensorDataset(x_train, y_train)

"""
pytorch에서 데이터 셋을 만들어야 데이터 로더 클래스 사용 가능
batch_size는 통상적으로 2의 배수를 사용 (2, 4, 8, 16..)
shuffle: 데이터를 섞어서 가지고 오는 역할 (일반적으로 True로 셋팅)

"""
# batch size 4개중에서 2개씩 끊어서 봄
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = nn.Linear(4, 1)    # => 4x1
optimizer = optim.SGD(model.parameters(), lr=1e-6)

# 모든데이터가 한번 돌면 => Epoch 1
nb_epoch = 5
for epoch in range(nb_epoch+1):
    # for batch_idx, (data, target) in enumerate(dataloader):
    # samples == data, target
    for batch_idx, samples in enumerate(dataloader):
        print(samples)
        x_train, y_train = samples
        # H(x) ==> wx + b (4x3 matmul 3x1 + bias)
        prediction = model(x_train)
        # loss 계산
        loss = F.mse_loss(prediction, y_train)

        optimizer.zero_grad() # 누적 미분으로 반드시 해줘야 함.
        loss.backward() # w1, w2, w3, b 각각 미분
        optimizer.step()  # w1~w3, b 업데이트 (w1 = w1-lr(미분값))

        print(f'Epoch {epoch} | Batch {batch_idx} | Loss {loss.item()}')

# 학습된 결과 확인(파라미터 값 확인)
# 파라미터: w1~w3, b
for param in model.parameters():
    print("-----------------")
    print(param.data)    # [w1, w2, w3] ==> b   (w1, w2, w3가 곱해짐)