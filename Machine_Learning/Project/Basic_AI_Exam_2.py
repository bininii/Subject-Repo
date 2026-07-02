### 2.
import torch

from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F

# 데이터셋 로드
mnist_train = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = dsets.MNIST(root='MNIST_data/', train=False, transform=transforms.ToTensor(),download=True)

# 데이터로더 설정
train_loader = DataLoader(mnist_train, batch_size=100, shuffle=True)

# Softmax Regression
model = nn.Linear(28*28, 10)
optimizer = optim.SGD(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# 학습
for epoch in range(5):
    for images, labels in train_loader:
        images = images.view(-1, 28*28)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

# 모델 저장
torch.save(model, 'mnist_softmax.pth')

# 모델 로드 후 평가 코드
model = torch.load('mnist_softmax.pth', weights_only=False)
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for images, labels in DataLoader(mnist_test, batch_size=100):
        images = images.view(-1, 28*28)
        outputs = model(images)
        max_val, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Accuracy: {accuracy:.4f}')


