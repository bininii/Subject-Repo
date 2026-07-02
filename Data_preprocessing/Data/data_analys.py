import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import seaborn as sns
df = sns.load_dataset('mpg')
print(df.head())
# mpg cylinders displacement horsepower weight
# acceleration model_year origin name

df = pd.read_csv(r'/Data/auto-mpg.csv', header=None)
print(df.head())
df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
              'acceleration', 'model_year', 'origin', 'name']
print(df.head())
# print(df.tail(3))
print(df.shape) # 연비를 예측하는 Linear Regression 모델 만들어보세요.
# nn.Linear(8, 1) ==> 연비는 정답 값(mpg->gt) ==> 398x8 ==> (398x8) matmul (8x1)
print(df.info())

#2. 데이터 요약 정보 확인하기
print(df.describe(include='all'))   # include: all, number, str  ## include='all' => 문자열 데이터도 같이 나옴!

# # 3. count, value_counts
# print(df['mpg'].count(), df.mpg.count())
# print(df['origin'].value_counts()) # value counts => dropna 옵션
# # uci 에서는 누락 데이터 없음
# import seaborn as sns
# titanic = sns.load_dataset('titanic')
# print(titanic.info())
# print(titanic['deck'].value_counts()) # dropna=True default  # dropna가 count하지 않은게 defalut임!
# print(titanic['deck'].value_counts(dropna=False))
#
# # value counts => group화 느낌. ==> 남자, 여자 생존 여부 그룹
# print(titanic[['survived', 'sex']].value_counts(dropna=False))
# # Boolean Filtering에서 한번 더 언급.
#
# # 4. 통계함수 (mean, median, etc.)
# # print(df.mean())
# print(df[['mpg', 'weight']].mean())
# print(df[['mpg', 'weight']].median())

##########################################
# 과제
## median filter / selected sort algorithm 구현해보기.
# def select_sort(in_data):
#     pass
# def median_filter(in_img):
#     select_sort(in_img[0:3, 0:3])
#     pass
# import cv2
# noise_img = cv2.imread('salt-pepper-noise.jpg')
# print(type(noise_img), noise_img.shape)
# input_img = cv2.cvtColor(noise_img, cv2.COLOR_BGR2GRAY)
# filter_img = median_filter(input_img)
# print(input_img.shape)
# cv2.imshow("noise_img", noise_img)
# cv2.waitKey()

#########################################
## 통계함수 적용 - 딥러닝 수학 corr(상관계수)
print(df['mpg'].std())
print("correlation between mpg and weight")
print(df[['mpg', 'weight']].corr(method="kendall"))
print("correlation between weight and displacement")  # 무게와 배기량의 관계
print(df[['weight', 'displacement']].corr())  # => corr 활용해볼것 !!
# print('correlation between mpg and all features')
# print(df.corrwith(df['mpg'], method='pearson'))

df.drop(['horsepower', 'name'], axis=1, inplace=True)
print('correlation between mpg and all features')
print(df.corrwith(df['mpg'], method='pearson'))









