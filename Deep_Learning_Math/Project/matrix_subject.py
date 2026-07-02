#Matrix Multiplication 코드 구현
#생성형 AI(ChatGPT 등)을 활용하지 않고 스스로 코드 작성해보기
#구현 Language(Java, Python, C, C++, etc.)는 상관없음
# for문 활용, 주석 작성


n1 = [[1, 2, 3], [4, 5, 6]]
n2 = [[1, 2], [3, 4], [5, 6]]
n3 = [[0,0], [0,0]]

for i in range(2):          #i: 0,1(n1의 행 개수)
    for j in range(2):      #j: 0,1(n2의 열 개수)
        for h in range(3):   #h: 0,1,2(n1의 열, n2의 행 개수)
            n3[i][j] += n1[i][h] * n2[h][j]

for row in n3:
    print(row)








