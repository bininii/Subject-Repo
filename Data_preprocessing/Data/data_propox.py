import pandas as pd
import seaborn as sns

df = sns.load_dataset('titanic')

# df = df.loc[:, ['age', 'fare']]
# print(df.head())

# 사용자 함수: 10을 더하는 함수
# 실습 시간에 좀 더 복잡한 함수를 만들어볼 것
def add_10(num):
    return num + 10

def add_two_param(a,b):
    return a + b

# sr1 = df['age'].apply(add_10)
# sr_add = df['age'] + 10
# print(sr1)
# print(sr_add)
#
# def min_max(x):
#     return x.max() - x.min()
#
# df_map = df.map(add_two_param, b=10)
# print(df.head())
# print(df_map.head())
#
# result = df.apply(min_max)
# print(result)
#
# df1 = pd.read_excel('stock price.xlsx')
# df2 = pd.read_excel('stock valuation.xlsx')
# print(df1.head())
# print(df2.head())
#
# # 교집합(merge: inner, all columns 대상)
# # how = inner, on: id, stock_name, value, price
# merge_inner = pd.merge(df1, df2)
# print(merge_inner)
#
# merge_outer = pd.merge(df1, df2, how='outer', on='id')
# print(merge_outer)
#
# # select * from df1 left outer join df2 on df1.stock_name=df2.name
# merge_left = pd.merge(df1, df2, how='left', left_on='stock_name', right_on='name')
# merge_right = pd.merge(df1, df2, how='right', left_on='stock_name', right_on='name')

print(df.head())
# group by 결과가 어떻게 나오는지 확인
df = df.loc[:, ['sex', 'age', 'class', 'fare', 'survived']]
print(df.head())

grouped = df.groupby(['class', 'sex'])
# print(grouped)
# print(grouped.sum())
# print(grouped.count())

for key, group in grouped:
    # print(f'*key: {key}')
    # print(f'*number: {len(group)}')
    # print(group.head())
    pass

# avg = grouped.mean()   # grouped.max()...etc.
# print(avg)

print(grouped.age.mean())
age_filter = grouped.filter(lambda x: x['age'].mean() <= 30)
print(len(age_filter[(age_filter['class'] == 'First') & (age_filter['sex'] == 'male')]))
print(len(age_filter[(age_filter['class'] == 'Second') & (age_filter['sex'] == 'male')]))
# print(age_filter)































