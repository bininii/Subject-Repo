import pandas as pd
import json

with open("../Data/000000.json") as f:
    data = json.load(f)
    df = pd.DataFrame(data['Object'])

    df_box = df['box2d'].apply(pd.Series)
    df = pd.concat([df, df_box], axis=1)

    # 1. class가 Dontcare인 데이터 삭제
    df.drop(df[df['class'] == 'Dontcare'].index, inplace=True)

    # 2. level이 0이 아닌 데이터는 1로 변경
    df.loc[df['level'] != 0, 'level'] = 1

    # 3. class가 car인 것은 Vehicle로 변경
    df.loc[df['class'] == 'Car', 'class'] = 'vehicle'

    # 4. class가 Truck이고 box2d의 넓이가 20000 이상이면 Bus로 변경
    box_size = (df['x2'] - df['x1']) * (df['y2'] - df['y1'])
    df.loc[(df['class'] == 'Truck') & (box_size >= 20000), 'class'] = 'Bus'

    # 5. x2, y2 ==> 에 width, height 값을 변경
    df.rename(columns={'x2': 'width', 'y2': 'height'}, inplace=True)

    print(df)
    dict_data = df.to_dict(orient='records')  # Object 형태로 저장 ==> [{}, {}]
    data['Object'] = dict_data

with open("../Data/김은빈_modified.json", "w") as f:  # w: write(작성하겠다)
    json.dump(data, f, indent=4)  # indent: 들여쓰기
        # ㄴ> 옵션 없을 경우 결과 확인해볼것
