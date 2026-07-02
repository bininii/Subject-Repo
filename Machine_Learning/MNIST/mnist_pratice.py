import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
data_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)

# 1. 모델 설계: 28x28 입력을 받아 10개의 클래스로 분류하는 3층 신경망
model = nn.Sequential(
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10) # 출력층: 10개의 클래스(숫자 0-9)로 분류
)

# 2. 손실 함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 3. 학습 루프(일부)
for data, target in data_loader:
    data = data.view(-1, 28 * 28)
    # (3) 기울기 초기화
    optimizer.zero_grad()

    output = model(data)
    loss = criterion(output, target)

    # (4)역전파 수행 및 가중치 업데이트
    loss.backward()
    optimizer.step()

# 4. 추론: 테스트 데이터셋의 첫 번째 이미지 예측
with torch.no_grad():
    test_img, label = mnist_test[0]
    test_img = test_img.view(-1, 28 * 28)
    prediction = model(test_img)

    # (5) 모델의 출력값(logits)에서 가장 확률이 높은 클래스의 인덱스 추출
    predicted_class = torch.argmax(prediction, dim=1)
    print(f"예측값: {predicted_class.item()}, 실제값: {label}")