##### 1. 데이터 split을 함수로 구현 (기존에 랜덤하게 70%는 train, 30%는 test)
# 함수의 인자로 train_test_split(아래) 메서드처럼 함수의 인자로 몇 %를 테스트 데이터로 할지 받아서 처리할 것

## 예시 ##
# 학습 및 평가를 위해 데이터 셋 분리 (k-fold 과제로도 수행 했었음)
x_train, x_test, y_train, y_test = (
    train_test_split(x, y, test_size = 0.3, random_state = 42)) # random_state: 랜덤한 데이터 추출할때 사용
# random state: 42 --> 0,1,2,3,4,5,7    test: 6,8,9  ==> random state에 지정(고정)한 숫자에 따라 test결과도 바뀜!
# random state: 77 --> 1,2,3,4,5,6,7    test: 0,8,9
## random seed == random state ##

##### 2. SVC 커널 변경, C, gamma 값 변경 등을 통해 성능 향상 시키기(2-1, 2-2: mandotory)
# 2-1) 입력 데이터 다양화(예: 성별, who 컬럼 등 포함해서 사용해보기)
# 2-2) svm 파라미터 변경
# 2-3) 그래도 잘 안나온다..? ==> 구글링 해보시기 바랍니다.
# 2-4) kaggle 순위 높은 방법론들 찾아서 github 한번 확인해보고 정리해보기
## 2-2까지 해보고 욕심나면 2-4까지 해보기
## 예시 ##
# 모델 선정 및 파라미터 셋팅
svm_model = SVC(kernel='rbf', C=0.1, gamma=10.0) # svc 모델 사용을 위한 기본적인 인자 셋팅

##### 3. svm_model.predict 결과를 활용하여
# - metrics.confusion_matrix 메서드 직접 구현해보기.
#  metrics.confusion_report 에서도 직접 구현.

## 예시 ##
# 학습된 모델 파라미터로 추론 및 10개 데이터 확인
y_pred = svm_model.predict(x_test)
# print(y_pred[0:10]) # [0, 0, 1, 0, 1]
# print(y_test[0:10]) # [1, 0, 0, 1, 1.]

from sklearn import metrics
svm_matrix = metrics.confusion_matrix(y_test, y_pred)
print(svm_matrix)
print(metrics.classification_report(y_test, y_pred))