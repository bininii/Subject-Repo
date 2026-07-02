import pandas as pd # pandas에 import
import seaborn as sns # matploylib -> 시각화 잘 하기 위한 용도 # seaborn -< sns로 축약
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
t_data = sns.load_dataset("titanic")
pd.set_option('display.max_columns', None) # 안보였던 컬럼에 대한 결과 볼수 있음 (None: 전체 다 보여줌)
# print(type(t_data))
print(t_data.head())
print(t_data.info())
# exit()   # 한줄한줄 실행하는거 보고싶을때 사용가능

# # ont=hot을 만들 때 pd.get_dummies 사용하면 됨.
# pclass_one_hot = pd.get_dummies(t_data.pclass)
# print(pclass_one_hot.head())
#
# # Boolean Indexing
# # df[(대괄호 안에 조건을 넣으면 됩니다.)]
# df_0 = t_data[(t_data['survived'] == 0)]
#
# df_1 = t_data[(t_data['survived']==1)]
#
# # survived == 1 and age >= 30
# df_2 = t_data[(t_data['survived']==1) & (t_data['age']>=30)]
# print(len(df_0), len(df_1), len(df_2))
#
# # concat : dataframe끼리 붙이기
# # df = pd.concat([t_data, pclass_one_hot], axis = 1)
# # axis 0/1 차이점 확인해볼 것
# df = pd.concat([t_data, pclass_one_hot], axis=1)
# print(df.tail())

df = t_data.drop(['deck', 'age', 'alive', 'class', 'embark_town'], axis=1) # drop: 행/열 데이터 삭제가능 * axis1= 열 기준 방향 삭제
# df의 embarked가 null인거 필터링
print(df[(df['embarked'].isnull())]) # 61, 829
# 최빈도 (embarked: S, Q, C)의 클래스를 비어있는 61, 829번에 입력
print(len(df[(df['embarked']=='S')]), len(df[(df['embarked']=='Q')]), len(df[(df['embarked']=='C')]))

df.loc[[61, 829], 'embarked']='S'  # df.iloc라면 -> df.iloc[[61, 829], 9]라서 가독성이 떨어짐. loc로 사용하는거 추천!
print(df[(df['embarked'].isnull())])    # None이 떠야함

# 여러분이 나중에 sex, who 컬럼의 값에 따라 str ==> int로 변경
"""
for idx, val in enumerate(df['who']):
    if val == 'man': df['who_int'] = 0
    if val == 'woman': df['who_int'] = 1
    if val == 'child': df['who_int'] = 2
"""

df.drop(['sex', 'who', 'embarked'], axis=1, inplace=True)
print(df.head())
# embarked <-- 카테고리니깐 "원-핫"으로 변경해서 입력 데이터로 사용해 볼 것
# 바로 위 까지의 결과 확인 원하면 exit() 입력하면 됨.

# 학습 데이터 준비
## x, y로 나누는이유? 입력 데이터, 출력데이터를 위해 지도학습 방법으로 준비.
# --> 우리는 지금 지도학습(:답을 가지고 스스로 학습하는 뱡향 ex)문제-답(답: "Ground Truth" -> y="GT")으로 생존여부를 판단하고자 함.
# vs 비지도학습(unsupervisual learning): 스스로 분류 ex) 거리기반으로 report 할때 사용
x = df.loc[:, 'pclass':'alone']
y = df['survived']

# 학습 및 평가를 위해 데이터 셋 분리 (k-fold 과제로도 수행 했었음)
x_train, x_test, y_train, y_test = (
    train_test_split(x, y, test_size = 0.3, random_state = 42)) # random_state: 랜덤한 데이터 추출할때 사용
# random state: 42 --> 0,1,2,3,4,5,7    test: 6,8,9  ==> random state에 지정(고정)한 숫자에 따라 test결과도 바뀜!
# random state: 77 --> 1,2,3,4,5,6,7    test: 0,8,9
## random seed == random state ##

# 모델 선정 및 파라미터 셋팅
svm_model = SVC(kernel='rbf', C=0.1, gamma=10.0) # svc 모델 사용을 위한 기본적인 인자 셋팅

# 인공지능 모델 학습
# fit 하고 나면 hyperplane: 도출 생존여부를 잘 판단할 수 있는 경계선을 찾았음
svm_model.fit(x_train, y_train)


# 학습된 모델 파라미터로 추론 및 10개 데이터 확인
y_pred = svm_model.predict(x_test)
# print(y_pred[0:10]) # [0, 0, 0 ...0] # 30% 뗀것중 10개 비교
# print(y_test[0:10]) # [1, 0, 0, 1, 1... 0]

from sklearn import metrics
svm_matrix = metrics.confusion_matrix(y_test, y_pred)
print(svm_matrix)
print(metrics.classification_report(y_test, y_pred))


# confusion Matrix("행/열"): classification 모델(SVM->SVC)의 성능평가시 사용하는 metrics("기준")
##  confusion Matrix: 모델 예측을 잘 못한 case가 얼마나 많이 나왔는지?
# ***Matrix-metrics 구분!!!!


