import pandas as pd

## sorted_index() 예시
'''dict_data = {'c0':[1,2,3], 'c1':[5,4,6], 'c2':[7,8,9]}
df = pd.DataFrame(dict_data)
# df = pd.Dataframe.from_dict(dict_data)
print(df)

sorted_df = df.sort_values(by='c1', ascending=False)
print(sorted_df)'''

## 산술연산
# 시리즈 + 숫자 연산
std1 = pd.Series({'kor':100, 'eng':80, 'math':90})
std2 = pd.Series({'eng':80, 'math':80})
addition = std1 + std2
addition = std1.add(std2, fill_value=0)
print(addition)
fill_val = 0
subtraction = std1.subtract(std2, fill_value=fill_val) #std1 - std2
multiplication = std1 * std2
# zero division 방지하기 위해서 입실론(굉장히 작은 값을 셋팅해서 사용)
division = std1 / std2 # 나누기 연산은 결과가 어떻게 될까요? std1.divide(std2, fill_value=0)
result = pd.DataFrame([addition, subtraction, multiplication, division],
                    index=['addition', 'subtraction', 'multiplication', 'division'])
print(result)

#유인물 6번 실습
#df1 = pd.Series({'row0, columnA':15,
#df2 = pd.Series({'row0, columnA':10})


