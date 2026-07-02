import pandas as pd

dict_data = {'c0': [1, 2, 3], 'c1': [4, 5, 6], 'c2': [7, 8, 9], 'c3': [10, 11, 12], 'c4': [13, 14, 15]}

# 행인덱스/열 이름 설정
df = pd.DataFrame(
[
    [23, '남', '서울'],
    [24, '여', '대구'],
    [22, '여', '대전']],
columns = ['name,', 'age', 'gender'],
index = ['홍길동', '신수민', '신수아'])

#print(type(df))
#print(df)
print(df.loc['신수아', 'gender'])
df.index = ['학생1', '학생2', '학생3']
df.columns = ['나이', '성별', '지역'] # columns --> 물론 영어로 하는걸 권장!
#print(df)
df_new_cols = df.rename(columns={'나이':'age', '성별':'gender'})

###실습 시간에 해보세요
# df_new_cols = df.rename(index={'나이':'age', '성별':'gender'})
# print(df.columns)
# print(df_new_cols.columns)

df_new_cols = df.rename(index={'나이': 'age', '성별': 'gender'})
print(df.columns)
print(df_new_cols.columns)


## drop 메서드 실습
exam_data = {'AI': [90, 100, 95], 'Data_Structure': [80, 80, 90],
             'JAVA': [100, 95, 85], 'ML': [90, 80, 100]}
df = pd.DataFrame(exam_data, index=['학생1', '학생2', '학생3'])
print(df)
# df.drop(['학생'], inplace=True)
## 행 삭제 (axis =0)
df_1 = df.drop(['학생1'], axis=0)  # axis=0: default --> 표기 하지 않아도 됨
print(df_1)
df_2 = df.drop(['학생1', '학생2'])
print(df_2)

### exam_data를 데이터프레임 만들때 인덱스를 지정하지 않고 생성 후 drop 시 인덱스 번호를 이용해서 삭제해보세요. (실습시간)
# ㄴ 학생 1, 2가 아니라 그냥 0,1,2로 해보기
# ex: df.drop([0,1])
exam_data = {'AI': [90, 100, 95], 'Data_Structure': [80, 80, 90],
             'JAVA': [100, 95, 85], 'ML': [90, 80, 100]}
df = pd.DataFrame(exam_data, index=[0, 1, 2])
print(df)


## 열 기준 삭제
df_3 = df.drop(['AI', 'Data_Structure'], axis=1) ## 열(column)단위 삭제는 반드시 명시!! (axis=1)
print(df_3)

### drop 시 슬라이싱으로 삭제도 해보세요.  --> 'dataname slicing drop' 구글링으로 참고하는 방법도 ㄱㅊ
# ex: df.drop([0:2]) 삭제
#1
df_ndf = df.drop(df.index[0:2])
#2
exam_data = {'AI': [90, 100, 95], 'Data_Structure': [80, 80, 90],
             'JAVA': [100, 95, 85], 'ML': [90, 80, 100]}
df.drop(df.index[0:2])
print(df)

# ex: df.drop(['AI': 'JAVA'])
#1
df1 = df.drop(columns=['AI', 'JAVA'])
print(df)
df = pd.DataFrame(exam_data, index=['학생1', '학생2', '학생3'])


data_structure = df['Data_Structure']
# data_structure = df.Data_Structure
# print(type(data_structure))
# print(data_structure)
# ai_ml = df[['AI', 'ML']]
# print(ai_ml)
point_std1_ml = df.loc['학생1', 'ML']
print(point_std1_ml)
print(df)
points_std1 = df.iloc[0, [2, 3]]
print(points_std1)
points = df.loc[['학생1', '학생3'], ['JAVA', 'ML']]
points = df.loc['학생1', 'AI':'JAVA']
print(points)
print(type(points))

# iloc, loc --> 실습! 여기도 슬라이싱 적용해볼 것
# 익숙해지도록 연습하는거 추천

df['WEB'] =[80, 90, 95]
print(df)

df.loc['학생4'] = 0
print(df)
df.loc['학생5'] = [100, 95, 85, 90, 90]
print(df)
df.iloc[3, 0] = 80 # 원소 접근은 iloc, loc but, 추가할때는 loc
print(df)
df.loc['학생6'] = df.loc['학생4']
print(df)
df['Eng'] = (df['AI'] + df['Data_Structure'] + df['JAVA'] + df['ML'])/4
print(df)

# 원소 값 변경
df.iloc[3, 0] = 80
df.iloc[5, [1, 2, 3]] = [80, 90, 87]

# slicing 으로 데이터 변경 (실습)
#df.iloc[3, 2:]
# df.loc['학생4' , :] = []
print(df)

