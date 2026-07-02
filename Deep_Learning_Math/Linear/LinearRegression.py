import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy.f2py.crackfortran import lenarraypattern
from sympy.vector import gradient


# 함수 작성 시 comment를 다는 습관을 가지자 !!
###################################################
# 함수명: get_data
# input: None
# output/return: train_x, train_y (Type: Torch Tensor)
# 선형 방정식 y = ax+b를 바탕으로 (a: gradient(기울기), b: intercept(절편))
# 오차를 포함한 x, y 데이터 생성 후 생성된 데이터 반환
###################################################
def get_data():
    random.seed(424)
    # 선형 방정식 계수(Coefficient) 임의 지정
    grad = 2.3  # gradient
    intercept = 0.88

    # 오차 범위: 10%
    gap = 0.1

    # number of Data: 20
    num_data_points = 20

    x_values = []
    y_values = []

    # x , y data 생성
    for i in range(num_data_points):
        #1: x값[1, 20] 중에서 랜덤으로 하나 추출
        x = random.uniform(1, 20)  # 1~20사이에서

        #2: 직선방정식 + 랜덤 오차
        linear_eq = grad * x + intercept
        noise = random.uniform(1-gap, 1+gap)
        y = linear_eq * noise

        x_values.append(x)
        y_values.append(y)

    # list => numpy => torch tensor 로 형 변환
    train_x = torch.from_numpy(np.array(x_values)).view(num_data_points, 1)
    train_y = torch.from_numpy(np.array(y_values))

    return train_x, train_y

###################################################
# 함수명: get_weights
# input: None
# output/return: 학습파라미터인 w(weights), b(bias)
# 학습 파라미터를 랜덤으로 하나 지정
# requires_grad + True 셋팅 (w, b 모두)
###################################################
def get_weights():
    w = torch.randn(1)  # w 한개만 가져옴
    b = torch.randn(1)  # b 한개만 가져옴

    w.requires_grad = True
    b.requires_grad = True

    return w, b

###################################################
# 함수명: simple_network
# input: train_x
# output/return: 학습파라미터인 w(weights), b(bias)가 반영된 예측 값
# H(train_x) = w*train_X + b   ==> 이걸 구해야함 !!
# torch.matmul 사용? for문 사용? 뭘 사용할지 결정해서 코드 작성해보세요. * matmul: Matrix Multiplication(행렬 곱셈)
###################################################
def simple_network(x): # 카멜: simpleNetwork, trainX, lossFunc
    y_pred = torch.matmul(x.float(), w.float()) + b
    # x.float => 20, w.float => 1
    # train_x = torch.from_numpy(np.array(x_values)).view(num_data_points, 1) ==> 이 부분 .view~ 이렇게 바꿔야함!!
    # train_y = torch.from_numpy(np.array(y_values))
    return y_pred

# 과제2: loss/cost function 등등의 함수에 대해 주석 작성해보기
############

############  ==> 여기 안쪽 부분!
def loss_func(y_pred, y):
    loss = torch.mean((y_pred - y).pow(2).sum())
    # 누적 미분 방지
    for param in [w, b]:
        if param.grad is not None:
            param.grad.data.zero_()  # 미분을 한번이라도 했으면 data를 초기화하라는 뜻.

    loss.backward()
    return loss.data                     # loss값 tensor이기 때문에 loss.data라고 명시해줘야 함

def param_update(lr):
    w.data = w.data - lr * w.grad.data    # w가 tensor이기 때문에 w.data라고 명시해줘야 함
    b.data = b.data - lr * b.grad.data



if __name__ == '__main__':
    # 1. 학습 데이터 준비
    train_x, train_y = get_data()
    print(train_x.shape)
    print(train_y.shape)

    # 2. 학습 파라메터 셋팅
    w, b = get_weights()
    print(w.shape, b.shape)
    print(w, b)
    # epoch 을 돌면서 수행할 함수
    # 과제1: 네임스페이스를 자동으로 찾아가는 방식이 안되도록 함수 작성해볼 것
    ## 최대한 지역변수와 return 값을 활용해서 함수 작성해볼 것
    # 전역변수 => 꼭 필요할떄만 사용, 지역변수 사용해서 구현할것
    # simple_network(train_x, w, b)
    # 3. 하이퍼파라메터 셋팅
    learning_rate = 1e-6   # ==> 10^-6(10의 -6승)
    num_epochs = 1000

    # 그래프 그릴 때 필요한 값 지정
    loss_x = []
    loss_y = []

    # 4. 지정된 epoch 만큼 돌면서 경사타고 내려오는 거.. 구현
    for epoch in range(num_epochs+1): ## 제일 마지막에 떨어지는게(0~1000) 1000번째! => 제일 마지막에 떨어지는거 보려고 +1이라고 붙임
        # 가설함수 "예측 값" (첫번째로 예측한 값)
        # 우리는 데이터를 20개만 사용했기 때문에 20x1의 사이즈의 데이터를 예측
        y_pred = simple_network(train_x)

        # loss 값 계산 및 미분
        loss = loss_func(y_pred, train_y)
        if epoch % 50 == 0:
            print(f'Current Loss: {loss}')
        loss_x.append(epoch)
        loss_y.append(loss)

        # 학습 파라미터 갱신
        param_update(learning_rate)

    #5. 결과물 시각화 (pytorch #2 pdf파일 슬라이드 9번 그래프)
    plt.figure()
    # 데이터 분포와 추론된 모델의 파라미터로 그은 선
    plt.plot(train_x.detach().numpy(), train_y, 'ro', label='Data')  # 색은 red, 동그라미로 표현하겠다.
    plt.plot(train_x.detach().numpy(), y_pred.detach().numpy(), 'b', label='prediction')
    plt.legend()
    plt.title("Linear Regression")
    plt.show()

    plt.figure()
    # loss 값이 안정적으로 잘 떨어지는지 확인
    # learning_rate를 더 작게할지 크게할지 결정 가능
    plt.plot(loss_x, loss_y)
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.show()




