## 데이터 전처리
## 데이터 살펴보기_ page.21
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False ## 3,4번째줄 한글 깨질때 복붙해서 사용할 것!! ##

# df = pd.read_csv('sell_bike.csv',
#                  encoding='cp949', index_col='자전거매장')
# df = pd.read_csv('sell_bike_inc_kinds.csv',
#                  encoding='cp949', index_col='자전거매장')
# print(df)
# x축: 자전거매장, y축: Data, 범례: column(1열~6열)
# df.plot(kind='bar')
# plt.show()

# index = ['yes' , 'no']
# print(df[df >= 450].count().sum())  # count => True 개수만 count!
# print(df[df < 450].count().sum())

# yes_cnt = df[df >= 450].count().sum()
# no_cnt = df[df < 450].count().sum()
# print(yes_cnt, no_cnt)
# y_data = [yes_cnt, 0]
# plt.barh(index, y_data, color='orange', height=0.6)
# y_data = [0, no_cnt]
# plt.barh(index, y_data, color='brown')
# plt.show()
#
# index = [1, 2, 3, 4, 5, 6]
# London = list(df.loc["런던"].values)
# York = list(df.loc["요크"].values)
# # print(London)
# L_450 = []
# for val in London:
#     if val > 450:
#         L_450.append(val)
#     else:
#         L_450.append(0)
#
# #요크는 list 중점으로 한번 구현 ..
# Y_450 = [val if val >= 450 else 0 for val in York]
# print(Y_450)
#
# plt.subplot(2, 1, 1)
# plt.bar(index, London, color='brown')
# plt.bar(index, L_450, color='orange')
# plt.legend(['No', 'Yes'], loc='best')
# plt.axhline(y=450, xmin=0, xmax=1, color='blue', linestyle='dashed') # min, max: 0, 1 (0~1 사이 비율로 작성)
# plt.ylabel('London')
# plt.ylim(0, 1200)
#
#
# plt.subplot(2, 1, 2)
# plt.bar(index, York, color='brown')
# plt.bar(index, Y_450, color='orange')
# plt.axhline(y=450, xmin=0, xmax=1, color='blue') # min, max: 0, 1 (0~1 사이 비율로 작성)
# plt.ylabel('York')
# plt.ylim(0, 1200)
#
# plt.show()

# print(df)

## page.24 데이터 살표보기 - 막대그래프 예시 (1)
# df_sum = df.sum(axis=1)
# print(df_sum)
# df_sum.sort_values(axis=0, inplace=True)
# df_sum.plot(kind='barh', grid=True)
# # df_sum.plot.barh()
# plt.xlim(2200, 3500)
# plt.xticks(list(range(2200, 3500, 100)))
# plt.show()

## page 25. 막대그래프 예시 (2)
# 각매장에서 어떤 종류의 자전거가 얼마나 주문되었나 or 어느 매장이 각 종류의 자전거를 가장 많이 주문?
# df.plot.barh(color=["#000000", "#555555", "#AAAAAA"], stacked=True)  # rgb color를 뜻함 ㅣ ex) 00-r , 00-g, 00-b
# plt.show()
#
# df_trans = df.T
# print(df_trans)
# df_trans.plot(kind='barh', color=["#000000", "#555555", "#AAAAAA"], stacked=True)
# plt.show()

## page 26.
# ax = plt.subplot(2, 1, 1)
# df.plot.barh(ax=ax, color=["#000000", "#555555", "#AAAAAA"], stacked=True)
# ax= plt.subplot(2, 1, 2)
# df_trans.plot(ax=ax, kind='barh', color=["#000000","#AAAAAA"], stacked=True)
# plt.show()

# fig, axes = plt.subplots(nrows=2, ncols=1)
# df.plot.barh(ax=axes[0], color=["#000000", "#555555", "AAAAAA"], stacked=True)
# df.trans.plot(ax=axes[1], kind='barh', color=["#000000", "#555555", "AAAAAA"], stacked=True)
# plt.show()
#

# # df = pd.read_excel('남북한발전전력량.xlsx')
# # df.info()
# # print(df.head(10))
# #
# # # column: 1990~2016 => 범례(라벨)
# # # 인덱스(row): x축
# # # value: y축
# # df_sn = df.iloc[[0, 5], 2:]
# # print(df_sn)
# # # df_sn.plot()
# # # plt.show()
# # df_vis = df_sn.T
# # # print(df_vis)
# # df_vis.columns = ['south', 'north']
# # df_vis.plot()
# # df_vis.plot(kind='bar')
# # df_vis.plot.barh()
# # plt.show()
#
# df = pd.read_csv('auto-mpg.csv', header=None)
# # print(df.head())
# df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
#               'acceleration', 'model_year', 'origin', 'name']
# # df_mpg = df['mpg']
# # df_mpg.plot(kind='hist', bins=8)
# # plt.show()
# # df.plot(kind='scatter', x='weight', y='mpg')
# # plt.show()
#
# df_box = df[['mpg','cylinders']]
# df_box.plot(kind='box')
# plt.show()
#

df = pd.read_excel('시도별 전출입 인구수.xlsx')


## ffill / bfill / fillna 기억해둘 것!! => 데이터 분석 / 처리 할 때 사용
# ffill(Forward Fill): NaN 나오기 바로 직전의 데이터로 NaN 칸을 채우는 방법
# bfill(Backward Fill): 거꾸로 올라가면서 채우는 방식
# df = df.fillna(method='ffill')
df = df.ffill()    # df.bfill()

# 전출지가 서울이면서 전입지가 경기도인 데이터만 추출
# 서울 => 경기도
mask = (df['전출지별'] == '서울특별시') & (df['전입지별'] != '서울특별시')
# print(mask.head(50))
df_s = df[mask]
print(df_s)
df_s = df_s.drop(['전출지별'], axis=1)
df_s.set_index('전입지별', inplace=True) # df_s = df_s.set_index('전입지별')
print(df_s)

sr_g = df_s.loc['경기도']
# print(sr_g)
# print(type(sr_g))
# plt.plot(sr_g)
# plt.show()
#
print(plt.style.available)
plt.style.use('ggplot')

# 그래프 꾸미기
# 사이즈 지정, 연도 기울이기, 선 옵션 넣기
# plt.figure(figsize=(14, 6))   # inch, 1 inch = 2.5cm
# # x축 회전
# plt.xticks(rotation='vertical')   # 90
# plt.plot(sr_g, marker='o', markersize=5, color='olive')
# plt.title('서울 --> 경기도 이동 인구 추이')
# plt.xlabel('기간(연도)')
# plt.ylabel('이동 인구수')
# # 범례지정
# plt.legend(loc='best', labels=['서울->경기'])
# plt.ylim(50000, 800000)  # annotate / legend 공간 확보
# # 그래프에 화살표 및 주석 넣기 (annotate)
# # annotation - 화살표
# plt.annotate('',
#             xy=(20, 600000),
#             xytext=(2, 300000),
#             arrowprops=dict(arrowstyle="->", color="skyblue", lw=4)
# )
# plt.annotate('',
#             xy = (48, 500000),
#             xytext = (30, 600000),
#             arrowprops = dict(arrowstyle="->", color="olive", lw=4)
# )
# # annotation - 그래프에 텍스트 삽입
# plt.annotate('인구증가 구간',
#             xy = (10, 500000),  # 텍스트 위치 기준점
#             xytext = (2, 300000)
#
# )
#
# plt.annotate('인구증가 구간',
#              xy=(10, 600000),
#              xytext=(30, 600000),
#             arrowprops = dict(arrowstyle="->", color="olive", lw=4)
#              )
#
# # annotation - 그래프에 텍스트 삽입
# plt.annotate('인구증가 구간',
#             xy = (10, 500000),  # 텍스트 위치 기준점
#             rotation =30,
#             va='center',
#             ha='center',
#             fontsize=12
#              )
# plt.annotate('인구감소 구간',
#             xy = (40, 550000),  # 텍스트 위치 기준점
#             rotation =20,
#             va='center',
#             ha='center',
#             fontsize=12
#              )
# plt.show()

# fig = plt.figure(figsize=(10, 10))
# ax1 = fig.add_subplot(2, 1, 1)  # row, col, index
# ax2 = fig.add_subplot(2, 1, 2)  # row, col, index
#
# # ax1,2 객체에 plot 함수 셋팅
# ax1.plot(sr_g, 'o', markersize=8, color='red')
# ax2.plot(sr_g, marker='o', markersize=8, markerfacecolor='green',
#          color='olive', lw=2, label='seoul->gyeoungido')
# ax2.legend(loc='best')
# ax1.set_ylim(50000, 800000)
# ax2.set_ylim(50000, 800000)
# # ax1.set_xticks(sr_g.index)
# # ax1.set_xticklabels(sr_g.index, rotation=70)
# # ax1.set_xticklabels(sr_g.index)
# ax1.tick_params(axis='x', labelrotation=70)
# plt.show()

#p.42 왼쪽 그래프 실습
# fig = plt.figure(figsize=(12, 5))
# ax1 = fig.add_subplot(1, 1, 1)
# df_cities = df_s.loc[['충청남도', '경상북도', '강원도'], :]
# print(df_cities.head())
# # ax1.plot(x축: 1970~2017)
# years = list(map(str, range(1970, 2018)))
# ax1.plot(years, df_cities.loc['충청남도', :], marker='o',
#          markerfacecolor='green', markersize=8, color='olive', lw=2, label='서울->충남')
# ax1.plot(years, df_cities.loc['경상북도', :], marker='o',
#          markerfacecolor='red', markersize=8, color='red', lw=2, label='서울->경북')
# ax1.legend(loc='upper right')
# ax1.set_title('서울->지역 이동 인구수')
# plt.show()

"""
# 38 페이지 아래, 위 그래프 꾸며보기
# 42 페이지 : 강원도 포함해서 3개 그래프 시각화, 연도, x축 셋팅, x,y축 라벨 셋팅 등
# 42 페이지: add_subplot(2, 2, 1) -> (2, 2, 2) -> 자료와 최대한 동일하게 작성해볼 것

"""
### p.39 그래프
# df = pd.read_excel('남북한발전전력량.xlsx')
# print(df)
# ### 시각화를 위한 데이터 전처리 및 컬럼 생성
# df = df.loc[5:]
# df.drop('전력량 (억㎾h)', axis=1, inplace=True)
# df.set_index('발전 전력별', inplace=True)
# df = df.T
# print(df)
#
# # 증감율: [(현재년도-전년도)/전년도]*100
# # 합계: 현재년도
# df.rename(columns={'합계':'현재년도'})  # 실습 시간에 해보세요.
# df['전년도'] = df['합계'].shift(1)
# df['증감율'] = ((df['합계']/df['전년도']) -1) * 100
# print(df.head())
#
# ### 그래프 그리고 셋팅 등등
# ax1 = df[['수력', '화력']].plot(kind='bar', stacked=True)
# ax2 = ax1.twinx()
# ax2.plot(df.index, df['증감율'], marker='o', color='green')
"""
ax1, ax2 y축 범위 지정
ax1, ax2 x, y label 지정 (ax.set_ylabel)
ax2: 라인스타일을 점선으로 변경해보세요.. linstyle='-----'
그래프의 제목 세팅. 범례위치 지정 또는 best,,

"""
# plt.show()

### p. 40 그래프 실습
df = pd.read_csv('auto-mpg.csv', header=None)
# print(df.head())
df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
              'acceleration', 'model_year', 'origin', 'name']
print(df.info())
print(df.head())
# df['cnt'] = 1
# df_origin = df.groupby(['origin']).sum()
# print(df_origin)
# df_origin.index = ['USA', 'EU', 'Japan']
# df_origin['cnt'].plot(kind='pie', figsize=(8, 8),
#                       autopct='%.2f%%',
#                       colors=['blue', 'green', 'red'])
# plt.show()
"""
타이틀, 범례, 색상 등 조정해서 꾸며볼 것
"""

### p.44 seaborn 그래프
import seaborn as sns
df_scatter = df[['weight', 'mpg']]
sns.regplot(x='weight', y='mpg', data=df_scatter,
            scatter_kws={'color': 'green'}, line_kws={'color': 'red'})
plt.show()
"""
seaborn doc. 참고해서 좀 더 꾸며볼 수 있으면 꾸며볼 것
"""

flights_data = sns.load_dataset('flights')
df = flights_data.pivot(index='month', columns='year', values='passengers')
ax = sns.heatmap(df, cmap='viridis', annot=True, fmt='d')
plt.show()