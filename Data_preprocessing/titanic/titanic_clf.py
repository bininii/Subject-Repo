# 1.기계학습 프로그래밍 과제#2에서 수행한 데이터 분할 함수를 from sklearn.model_selection import train_test_split 에서
# train_test_split 대신 사용해서 구현해 볼 것
# from 파일명 import 메서드명 ← 사용해서 확인해볼것
# 예: 기계학습 프로그래밍 과제#2의 파일명 이름이 split_dataset.py 이고
# 함수 이름이 split_train_test (def split_train_test) 라고 가정하면
# 해당 과제 시작에 from split_dataset import split_train_test로 사용 가능


# 2. 모델 성능 향상해보기: 제공된 기초코드의 성능을 높이기 위해 아래사항을 수행해볼것
# - 입력데이터 다양화(FeatureEngineering)
# - 기존에 제외했던 sex, who, embarked 컬럼을 학습에 포함해볼것
# - 문자열 데이터를 숫자형으로 변경하고 필요시 원 - 핫인코딩으로 변경해서 사용할 것
# - SVM 하이퍼 파라미터 튜닝
# - kernel: rbf외 linear, poly등이있음
# - C, gamma값 조정 해볼것


# 3. 성능 평가 지표 직접 구현
# - `sklearn.metrics`에서 제공하는 라이브러리 결과를 확인하는 것에 그치지 않고,
#    예측값(`y_pred`)과 실제값(`y_test`)을 비교하여 Confusion Matrix 직접 구현해보기
# - Classification Report 결과인 precision, recall 만 구현해보기


# 제출 방법: 데이터 분할 함수를 포함하여 2개의 파이썬 파일 제출
# - split_dataset.py
# - titanic_clf.py (1~3번까지를 구현한 파일)


import pandas as pd
import seaborn as sns
import numpy as np

# 데이터 전처리
df = sns.load_dataset("titanic")
# 결측치 처리
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df['age'] = df['age'].fillna(df['age'].mean())

# sex, who, embarked 칼럼 포함
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'who', 'embarked']
X_data = df[features]

# 문자열 데이터를 one-hot encording(숫자형)으로 변경
X = pd.get_dummies(X_data, columns=['sex', 'who', 'embarked'], drop_first=True)
y = df['survived']

from split.split_dataset import split_train_test
x_train, y_train, x_test, y_test = split_train_test(X, y, test_size=0.3)


# 3. SVM 모델 하이퍼 파라미터 튜닝
# kernel  => rbf 모델
# C, gamma 조정
from sklearn.svm import SVC
svm_rbf = SVC(kernel='rbf', C=10, gamma=0.01)
svm_rbf.fit(x_train, y_train)
y_pred_rbf = svm_rbf.predict(x_test)


# 3. 성능 평가 지표 직접 구현
# Confusion Matrix
def get_metrics(y_test, y_pred):
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    tp = np.sum((y_test == 1) & (y_pred == 1))
    tn = np.sum((y_test == 0) & (y_pred == 0))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == 0))

    return tp, tn, fp, fn

# 함수 호출
tp, tn ,fp, fn = get_metrics(y_test, y_pred_rbf)

# Confusion Matrix 출력
print("-----------------------------------")
print(f"Confusion Matrix:\n [[{tn} {fp}]\n  [{fn} {tp}]] ")


# precision, recall 구현
def precision(tp, fp):
    if (tp + fp)  == 0:
        return 0
    return (tp) / (tp + fp)


def recall(tp, fn):
    if (tp + fn)  == 0:
        return 0
    return (tp) / (tp + fn)



# 결과 출력
precision_val = precision(tp, fp)
recall_val = recall(tp, fn)

print("------ Classification Report ------")
print(f"Precision: {precision_val:.4f}")
print(f"Recall   : {recall_val:.4f}")
print("-----------------------------------")

