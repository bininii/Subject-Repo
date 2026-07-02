import pandas as pd

# 1. Read CSV
# path = "Data/read_csv_sample.csv"
# path = "./Data/read_csv_sample.csv"  # 상대경로
# path = "../../Data/read_csv_sample.csv"


# file_path = "Data/read_csv_sample.csv"
# #
# # # header 정보를 생략해도 default: header = 0
# # df1 = pd.read_csv(file_path, header=0)
# # print(df1)
# #
# # # header = None option: row, col index를 자동으로 생성
# # df2 = pd.read_csv(file_path, header=None)
# # print(df2)
# #
# # # header = None option: row, col index를 자동으로 생성
# # df2 = pd.read_csv(file_path, header=1)
# # print(df2)
# # df = pd.read_csv(file_path, index_col='c0')
# # print(df)
#
#

# # 2. Read Excel
# # option (header, index_col,...,) : read_csv와 거의 동일
#
#
# df = pd.read_excel("./Data/남북한발전전력량.xlsx")  # header=0
# print(df.head())
# # header=none, 1,...



# df = pd.read_jason("Data/read_json_sample.json")
# print(df)

import json

# f = open('./data/000000.json')
# data = json.load(f)
# # print(data['Object'])
# df = pd.DataFrame(data['Object'])
# print(df.head())
# # class == 'Dontcare' 이거나 level != 0 많은 데이터는 삭제
# # class가 Truck 이거나 car 이면 Vehicle로 변경
# #  ==> boolean indexing, Filtering
# df.drop(df[(df['class'] == 'Dontcare') | (df['level'] != 0)].index, inplace=True)
#
# # class가 True 이거나 car 이면 Vehicle로 변경 ==> 여러분들이 직접 한 번 해보세요.
# dict_data = df.to_dict(orient='records')  # Object 형태로 저장 ==> [{}, {}]
# data['Object'] = dict_data
# f.close()

with open("../Data/000000.json") as f:
    data = json.load(f)
    df = pd.DataFrame(data['Object'])
    df.drop(df[(df['class'] == 'Dontcare') | (df['level'] != 0)].index, inplace=True)
    dict_data = df.to_dict(orient='records')  # Object 형태로 저장 ==> [{}, {}]
    data['Object'] = dict_data



with open("../Data/000000_modified.json", "w") as f:   # w: write(작성하겠다)
        json.dump(data, f, indent=4)   # indent: 들여쓰기
        # ㄴ> 옵션 없을 경우 결과 확인해볼것


# 제출파일 이름: {name}_modified_jsonFile.json
# 1. class가 Dontcare인 데이터는 삭제
# 2. level이 0이 아닌 데이터는 1로 변경
# 3. Class가 car인것은 Vehicle로 변경
# 4. class가 Truck이고(and) box2d의 넓이가 20000 이상이면 Bus로 변경
# 5. x2, y2 ==> 에 width, height 값을 변경




data1 = {
    'name' : ['Jerry', 'Riah', 'Paul'],
    'algo' : ['A', 'A+', 'B'],
    'c++' : ['A', 'A', 'B']
}
data2 = {
    'c0' : [1, 2, 3],
    'c1' : [4, 5, 6],
    'c2' : [7, 8, 9]
}
df1 = pd.DataFrame(data1)
# print(df)
# df.to_csv('data.csv')
df2 = pd.DataFrame(data2)

exc_writer = pd.ExcelWriter('df_excel_writer.xlsx')
df1.to_excel(exc_writer, index=False, header=False, sheet_name='Sheet1')
df2.to_excel(exc_writer, index=False, header=False, sheet_name='Sheet2')
exc_writer._save()
exc_writer.close()