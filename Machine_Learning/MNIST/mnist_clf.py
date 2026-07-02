## MNIST: '손글씨 숫자 데이터 세트'
## 컴퓨터에게 0~9까지 손글씨 숫자를 보여주고, 이게 무슨 숫자인지 맞히도록 가르치는(학습)과정
## 데이터 로드 => 모델 정의 => 학습(반복) => 평가 (4단계)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.datasets as dset  # mnist, fashion mnist, 등 가지고 올 수 있음
import torchvision.transforms as transforms   # dataset => Tensor 변경, to Image
import time


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
f_mnist_train = dset.FashionMNIST(root='./binary', train=True,
                                  transform=transforms.ToTensor(), download=True)
mnist_train = dset.MNIST(root='./binary', train=True,     ##dset.MNIST: 0~9까지 손글씨 이미지 다운로드
                         transform=transforms.ToTensor(), download=True)

mnist_test = dset.MNIST(root='./binary', train=False,
                        transform=transforms.ToTensor(), download=True)

# DataLoader: 100개씩 묶어서(batch) 조금씩 공부하도록 정리하는 도구
data_loader = torch.utils.data.DataLoader(mnist_train,
                                          batch_size=100, shuffle=True)
f_data_loader = torch.utils.data.DataLoader(f_mnist_train,
                                          batch_size=100, shuffle=True)
transform=transforms.ToPILImage()
# print(mnist_train)
for i in range(2):
    img = transform(f_mnist_train[i][0])
    print(f_mnist_train[i][1])
    print(img.size)
    plt.imshow(img, cmap='gray')
    plt.show()
# exit()
# 객체 생성 => model(x) => w * x + b
# (28, 28) => 2차원 데이터로 표현 ==> 1차원으로 계산하기 위해 28x28의 값을 사용하면 됨.
# (28, 28) => 28x28의 1차원 배열 데이터와 같습니다.
model = nn.Sequential()      ## nn.Sequential(): 인공지능이 생각하는 '뇌'의 층을 쌓음
## nn.Linear(입력, 출력): 이전 단계에서 받은 정보를 다음 단계로 전달하는 연결 통로
## 28x28(=> 784 pixel)을 입력받아서, 정보 압축 => 마지막엔 10개(0~9 숫자) 중 무엇인지 확률을 내놓음
## nn.ReLU(): 정보 중에서 중요한 것만 골라내는 '필터' 역할
model.add_module('fc1', nn.Linear(28*28, 512)) # fulluy Connected Layer   # output
model.add_module('relu1', nn.ReLU())
model.add_module('fc2', nn.Linear(28*28, 256)) # fulluy Connected Layer #input
model.add_module('relu2', nn.ReLU())
model.add_module('fc3', nn.Linear(256, 10)) # fulluy Connected Layer
model = model.to(device)   # 모델의 객체는 cuda에서 작성
# F.cross_entropy()  # 차이점..
los_fn = nn.CrossEntropyLoss()  # softmax(wx+b)
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(epochs):
    #epoch for 문 하나 추가해야 함.
    for epoch in range(epochs):
        for data, gt in data_loader:
        # data shape: 28x28 ==> mm.Linear 의 입력 값과 동일하게 맞추어줘야 함
            data = data.view(-1, 28*28)   ## => 이미지(28x28 2차원)를 모델에 넣으려면 784개의 길쭉한 1차원줄로 펴야함!
            data = data.to(device)        ## device: 그래픽 카드(CUDA)가 있다면 그 힘을 빌려서 훨씬 빠르게 학습시키겠다는 뜻
            gt = gt.to(device)
            optimizer.zero_grad()     ## optimizer.zero_grad(): 이전 학습의 흔적(기울기)을 지움 (새롭게 배우기 위해!)
            y_hat = model(data)       ## y_hat = model(data): 모델이 숫자를 보고 "이건 3인 것 같아!"라고 예측
            loss = los_fn(y_hat, gt)    ## loss = los_fn(y_hat, gt): 정답(gt)과 모델의 예측(y_hat)을 비교해서 얼마나 틀렸는지 점수 매김(loss)
            loss.backward()           ## loss.backward() & step(): !!중요!! 틀린 만큼 모델의 연결강도(가중치)를 살짝 수정 => 정답 맞히게 고쳐나가는 과정
            optimizer.step()
        print(f'epoch {epoch}, train loss: {loss.item()}')

train(50)
torch.save(model, 'mnist_dnn.pth')  # model의 정보가 들어가있음. (=> weight 값 => 특정한 실수 값 ex) 0.07 ..)


# inference(추론) => 보통 실행이라고 하지 않음. (learning -> x)
model = torch.load('mnist_clf.pth', weights_only=False)

# total number of data: 1000
# corret: 900 => 900/1000 => 0.9
correct=0
print(len(mnist_test))
for i in range(len(mnist_test)):
    x_test = mnist_test[i][0]
    x_test = x_test.view(-1, 784)
    prediction = model(x_test)
    prob = F.softmax(prediction, dim=1)
    gt = mnist_test[i][1]
    if torch.argmax(prob, dim=1) == gt:     ## torch.argmax(prob, dim=1): 모델은 각 숫자에 대한 '확률'을 출력하는데
        correct += 1                        ## (ex. 0일 확률 10%, 3일 확률 80%...), 그 중 가장 높은 숫자를 정답으로 선택하는 함수
print(f'Accuracy: {(correct / len(mnist_test))*100}%')
# # 학습시킨 모델을 통해서 추론
# for i in range(5):
#     img = transform(mnist_test[i][0])
#     print(mnist_test[i][1])
#     print(img.size)
#     plt.imshow(img, cmap='gray')
#     plt.show()
#
#     # [i] 인덱스, [0]: 이미지데이터, [1] 라벨(GT)
#     x_test = mnist_test[i][0]
#     print(x_test.shape, x_test.dim, type(x_test))
#     # 형태 변환을 해주어야 함.
#     x_test = x_test.view(-1, 784)
#     prediction = model(x_test)
#     print(prediction)
#     prob = F.softmax(prediction, dim=1)
#     print(prob)
#     print(torch.argmax(prob, dim=1))  ## argmax: list 에서 가장 큰 값이 있는 인덱스를 반환하는 메서드

