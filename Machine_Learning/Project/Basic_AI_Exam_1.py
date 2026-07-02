### 1.
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 데이터셋 로드 & 전처리
titanic = sns.load_dataset('titanic')
df = titanic[['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']].dropna()
X = torch.FloatTensor(df.drop('survived', axis=1).values)
y = torch.FloatTensor(df['survived'].values).view(-1, 1)

# 훈련 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 파라미터 초기화
w = torch.zeros((5, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
optimizer = optim.SGD([w, b], lr=0.001)

# 학습
for epoch in range(3000):
    hx = torch.sigmoid(X_train.matmul(w) + b)
    loss = -(y_train * torch.log(hx + 1e-7) + (1 - y_train) * torch.log(1 - hx + 1e-7)).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Accuracy
with torch.no_grad():
    prediction = torch.sigmoid(X_test.matmul(w) + b)
    predicted_classes = (prediction >= 0.5).float()
    accuracy = (predicted_classes == y_test).float().mean()

print(f'Accuracy: {accuracy:.4f}')