import pandas as pd

##1. 상위 300까지 포켓몬들과 401~800위까지의 포켓몬들에서 전설의포켓몬(Legendary컬럼)의 숫자 차이
# df = pd.read_csv('Pokemon.csv')
# df_sorted = df.sort_values(by='Total', ascending=False)
#
# group1 = df.iloc[:400]
# group2 = df.iloc[400:]
#
# count1 = group1['Legendary'].sum()
# count2 = group2['Legendary'].sum()
#
# diff = abs(count1 - count2)
# print(f"Legendary 포켓몬: {diff}")


##2. Type 1 컬럼의 속성이 fire인 포켓몬들의 Attack의 평균 이상인 Water 속성의 포켓몬 수를 구하시오.
# fire_median_attck = df[df['Type 1'] == 'Fire']['Attack'].median()
# water_pokemon_count = df[(df['Type 1'] == 'Water') & (df['Attack'] >= fire_median_attck)].shape[0]
# print(f"fire attack 중앙값: {fire_median_attck}")
# print(f"water pokemon count: {water_pokemon_count}")


##3. 10살 단위 변환 =>  가장 많은 인원 가진 "나이대와 인원수"
# from pandas import value_counts
# df = pd.read_csv('bank.csv')
# df['age_group'] = df['age'] // 10  * 10
# themost_group = df['age_group'].value_counts().idxmax()
# print(f"가장 많은 인원 가진 나이대와 인원수 : {themost_group}대")


##4. 나이 25이상 29미만 고객중 housing 컬럼의 값이 yes인 고객의 수?
# result = df[(df['age'] >= 25) & (df['age'] < 30) & (df['housing'] == 'yes')]
# print(f"yes 고객 수: {len(result)}명")


## 5. index값 기준으로 내림차순 정렬했을때 상위 100개 데이터의 balancd값의 평균은?
# mean_val = df['balance'].mean()
# filtered_df = df[df['balance'] > mean_val]
#
# top_100_mean = filtered_df.sort_index(ascending=False).head(100)['balance'].mean()
# print(f"상위 100개 balance 평균: {top_100_mean}")


## 8. 성별이 Male인 환자들의 age의 평균값을 구하시오.
# df_stroke = pd.read_csv('healthcare-dataset-stroke-data.csv')
# mean_age_male = df_stroke[df_stroke['gender'] == 'Male']['age'].mean()
# print(f"Male 환자 평균 나이: {mean_age_male}")


## 9. bmi컬럼의 평균값을 구하시오.
# df_stroke = pd.read_csv('healthcare-dataset-stroke-data.csv')
# df_stroke['bmi'] = df_stroke['bmi'].fillna(df_stroke['bmi'].mean())
# print(f"bmi 평균값: {df_stroke['bmi'].mean()}")

