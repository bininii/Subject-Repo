# california 집값 예측
import torch
from sklearn.datasets import fetch_california_housing
import numpy as np

def split_train_test(x, y, test_size=0.2):
   num_samples = len(x)
   indices = torch.randperm(num_samples)

   train_size = int(num_samples * (1 - test_size))
   train_indices = indices[:train_size]
   test_indices = indices[train_size:]

   x_train = x[train_indices]
   y_train = y[train_indices]
   x_test = x[test_indices]
   y_test = y[test_indices]

   return x_train, y_train, x_test, y_test


housing = fetch_california_housing()
X = torch.tensor(housing.data, dtype=torch.float32)
y = torch.tensor(housing.target, dtype=torch.float32).view(-1, 1)

# 랜덤 시드 설정
torch.manual_seed(10)

# shuffle
num_samples = len(X) # 전체 데이터(20640개)
indices = torch.randperm(num_samples) # 0~20639 무작위로 섞음

train_size = int(num_samples * 0.7) # 20640*0.7 = 14,448개
train_indices = indices[:train_size]
test_indices = indices[train_size:]


x_train = x[train_indices]
y_train = y[train_indices]
x_test = x[test_indices]
y_test = y[test_indices]




k=5
n = int(len(train_indices) * 0.2) # 14,448개의 20%

data_1 = x_train[0 : n]
data_2 = x_train[n : 2*n]
data_3 = x_train[2*n : 3*n]
data_4 = x_train[3*n : 4*n]
data_5 = x_train[4*n : ]

foldList = [data_1, data_2, data_3, data_4, data_5]

i = 0
j = 0

val = []
train = []



for i in range(5):
    # i번째 폴드를 검증 데이터로 사용
    val = foldList[i]
    # i번째가 아닌 나머지 폴드들을 합쳐서 학습 데이터로 사용
    train_list = [foldList[j] for j in range(5) if j != i]
    train = torch.cat(train_list, dim=0)
    print(f"Fold {i + 1} -> 검증셋: {val.size()}, 학습셋: {train.size()}")
    print(val[:1,:], train[:1,:])


