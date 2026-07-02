# California Housing Dataset 분할 및 K-Fold 직접 구현
# 머신러닝에서 필수적인 **데이터 분할(Data Splitting)**과 **교차 검증(K-Fold Cross Validation)**을 라이브러리 함수에 의존하지 않고
# **직접 구현**함으로써 내부 동작 원리를 이해하는 것을 목표로 한다.
#데이터 셋
# **California Housing Dataset**
# `sklearn.datasets.fetch_california_housing()` 활용

# 1. 데이터 로드 및 tensor 변환 -> 데이터를 불러온 뒤 torch.Tensor로 변환

# 2. Train / Test Split 직접 구현 (70% / 30%)
#  - `train_test_split` 사용 금지 (구현 결과 참고용 가능)
#  **PyTorch 연산만 사용하여 구현**
# - 반드시 shuffle 수행 후 분할
# - random seed 설정
# 구현 아이디어
# 1) 전체 인덱스를 torch로 생성
# 2) `torch.randperm()`으로 shuffle
# 3) 70% 기준으로 index 분할
# 4) index 기반 slicing

# 3.K-Fold Cross Validation 직접 구현 (PyTorch)
# - K = 5
# - `KFold` 사용 금지  (구현 결과 참고용 가능)
# 각 fold마다 validation set 1개, 나머지는 train set
# fold 크기 균등 분할

from sklearn.datasets import fetch_california_housing  # california 집값 예측
housing = fetch_california_housing()
print(type(housing))
print(type(housing.data))
print(housing.data.shape) # numpy type, m x n matrix
cal_tensor = torch.FloatTensor(housing.data)



