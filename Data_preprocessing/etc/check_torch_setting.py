import torch
#
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))
# print(torch.cuda.device_count())
print(torch.__version__)
# Vector --> 1D(Dimensional) Tensor

# 금주의 평균 온도
temp = torch.FloatTensor([17.4, 20, 13.7, 17.4, 20, 13.7, 20])
print(temp.size())  #size -> 메서드
print(temp.dim())
print(type(temp))

# type 텐서라고 하더라도 리스트와 같이 인덱스, 슬라이싱 모두 가능
print(f'월, 화 평균온도는: {temp[0]}. {temp[1]} 입니다.')

#슬라이싱으로 화~목 까지의 온도를 출력하세요.
print()

# 2D Tensor == Matrix
# 4 x 3 martix
m = torch.FloatTensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]])
print(m.size())
print(m.dim())
# 슬라이싱, 인덱싱 등 실습해보세요.
# 첫 번째 차원의 모든 것(모든 rows와  두번째 차원(col)의 두번째 값들만)
print(m[:, 1])
#첫 번째 차원의 3번째 요소들과 두 번째 차원의 모든 것
print(m[:2])

#[[5,6], [8, 9]] <-- 이렇게 한 번 가지고 와보세요.

#t=[1, 2, 3, 4, 5]
#for i in range(len(t)):
#    print(t[i])

#for i in range(6):
#    print(t[i])

from sklearn.datasets import fetch_california_housing  # california 집값 예측
housing = fetch_california_housing()
print(type(housing))
print(type(housing.data))
print(housing.data.shape) # numpy type, m x n matrix
cal_tensor = torch.FloatTensor(housing.data)



cal_tensor = torch.from_numpy(housing.data)
print(cal_tensor.size())
print(cal_tensor.dim())
print(cal_tensor[:10, :])



#data([20640, 8]) 70%가 몇개의 행인지 계산

#train_x= data(70%, 7) 처음부터 70% 위치의 행까지 ->처음부터 7번째 열까지(:7)
#train_y=data(70%, -1) 처음부터 70% 위치의 행까지 -> 마지막 열 하나만(-1)
#test_x=data(30%,7) 70% 위치부터 마지막 행까지(나머지 30%) -> 처음부터 7번째 열까지(:7)
#test_y=data(30%, -1)

#####과제#####
#기본 -> !!!! Random(중복되지 않도록) 70% 가지고 와서 train_x
# ㄴ 나머지 30%가 test_x 구성

#1 중복되지 않게 70% 랜덤
#2 "K-folder Testing"


#####과제 예시#####
# 전체 데이터의 70%를 학습 데이터 셋을 활용하고
# 30%를 테스트 데이터셋으로 구성하여 평가를 진행하고자 한다.
# 위 내용을 반영하여 데이터 셋 구축을 하세요.
train_len = int(len(cal_tensor) * 0.7)
print(train_len)
train_x = cal_tensor[:train_len, :7] # 0번째 행부터 제일 마지막까지 train_len -1번째 행까지 (==> random하게 70% 가져오는게 과제)
train_y = cal_tensor[:train_len, 7:] # :찍기!
# train_y = cal_tensor[:train_len, -1]
test_x = cal_tensor[train_len, :7]
test_y = cal_tensor[train_len, 7]
print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)

#####random하게 갖고오기 예시#####
rand_list =  [1, 3, 7] # 중복되지 않은 행을 랜덤하게 가지고 왔어요
train_x_rand = cal_tensor[rand_list, :7]
print(train_x_rand.shape)