import torch
import torch.optim as optim

# x_data: 6x2
# y_data: 6x1
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# 6x2 matmul (2x1) ==> 6x1 ==> w size: 2,1
w = torch.randn((2, 1), requires_grad=True)
w = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([w, b], lr=1e-6)

nb_epoch = 3000
for epoch in range(nb_epoch+1):
    #pass
    # 1. hypothesis (sigmoid)
    temp_x = x_train.matmul(w) + b
    # hx = sigmoid(temp_x.detach().numpy())   # sigmoid 함수를 토치로 만들기
    # hx = torch.from_numpy(hx)
    hx = 1 / (1 + torch.exp(-temp_x))

    # 2. cost function (-y(log(h(x)) - (1-y)log(1-h(x))) (기계학습 전달 ppt-page.15 슬라이드 참고)
    loss = -(y_train*torch.log(hx)+(1-y_train)*torch.log(1-hx)).mean()

    # 3. gradient descent algorithm & update param
    optimizer.zero_grad()
    loss.backward()  # -(y_train*torch.log(hx)+(1-y_train)*torch.log(1-hx)).mean() 룰 미분적용
    optimizer.step()

    # 4. cost 값이 잘 떨어지고 있는지 확인
    if epoch % 100 == 0:
        print(f'epoch: {epoch}, loss: {loss}')

    # 학습된  w, b 출력 ( 모델 파라메터 )
    print(w, b)

    # 학습이 잘 되었으면 w, b 값을 sigmoid 함수에 적용하면
    # 앞에 3개는 0.5보다 작은 값, 뒤에 3개는 0.5보다 큰 값 => [0], [0], [0]  / [1], [1], [1]
    prediction = torch.sigmoid(x_train.matmul(w) + b)
    print(prediction)

    pred = []
    for i in list (prediction):
        if i > 0.5: pred.append(1)
        else: pred.append(0)
    print(pred)

    result = prediction >= torch.FloatTensor([0.5])
    print(result)

