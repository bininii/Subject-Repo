import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from titanic.titanic_clf import features

# 타이타닉 데이터 불러오기
df = sns.load_dataset("titanic")
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df['age'] = df['age'].fillna(df['age'].mean())

features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
x_data = df[features]

# One-Hot Encoding
x_titanic = pd.get_dummies(x_data, columns=['sex',  'embarked'], drop_first=True)
y_titanic = df['survived']

# 마지막 row 20개 => 평가용(test)로 분리 / 20개 제외한 나머지=> 학습용(train)
x_train_titanic = x_titanic.iloc[:-20]
y_train_titanic = y_titanic.iloc[:-20]

x_test_titanic = x_titanic.iloc[-20:]
y_test_titanic = y_titanic.iloc[-20:]

# SVM 모델 & Logistic Regression 모델
svm_model = SVC(kernel='rbf', C=10, gamma=0.01)
lr_model = LogisticRegression(max_iter=1000)

# 모델 학습
svm_model.fit(x_train_titanic, y_train_titanic)
lr_model.fit(x_train_titanic, y_train_titanic)

# 정답 예측
y_predict_svm = svm_model.predict(x_test_titanic)
y_predict_lr = lr_model.predict(x_test_titanic)

# 성능 비교
svm_correct = 0
lr_correct = 0

# 20개 데이터와 정답(y_test_titanic) 비교
for i in range(20):
    # SVM이 맞춘 개수 누적
    if y_predict_svm[i] == y_test_titanic.values[i]:
        svm_correct += 1
    # Logistic Regression이 맞춘 개수 누적
    if y_predict_lr[i] == y_test_titanic.values[i]:
        lr_correct +=  1

# 결과 출력
print(f"SVM: {svm_correct}/20 ({int((svm_correct/20)*100)}%)")
print(f"Logistic Regression: {lr_correct}/20 ({int((lr_correct/20)*100)}%)")