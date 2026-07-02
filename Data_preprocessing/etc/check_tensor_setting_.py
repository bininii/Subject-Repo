# import torch
#
# # 3D Tensor
# # 대표적인 예가 컬러 이미지
# # w, h, c (width, height, channel)
# # 1920x1080x3(r, g, b)
# from PIL import Image
# import numpy as np
#
# img = np.array(Image.open("./cat.1.jpg").resize((100, 100))) # ./ : 현재 디렉터리를 표현
# #img = Image.open("dataset/training_set/cats/cat.10.jpg)
# img_tensor = torch.from_numpy(img)
# print(img_tensor.size())
# print(img_tensor.shape)
# print(img_tensor.dim())
#
# # import matplotlib.pyplot as plt
# # plt.imshow(img)
# # plt.show()
#
#
# # 4-dimensional tensor
# # 폴더에 있는 여러 이미지를 텐서에 담으면 4d
# # 64x100x100x3
# datapath = "dataset/training_set/cats/"
# # 경로에 있는 파일명들을 리스트에 넣기
# # [datapath + cat1.jpg, datapath + cat2.jpg.....]
# from glob import glob # glob: data를 특정한 폴더(디렉터리)에 있는 것들을 긁어서 리스트화 하고 싶을때 사용
# # *.* 모든 확장자 + 모든 파일
# cats = glob(datapath + "*") # *: all (모든 변수를 의미함) => cats = "dataset/training_set/cats/*"
# print(cats[:2])
# img_list = []
# for cat in cats:
#     img_list.append(np.array(Image.open("./cat.1.jpg").resize((100, 100))))
#
# # ==> 리스트 중첩으로 한번 해보세요
#
# # [ [img1-3d array], [img2-3d array],...,[img-n-3d array] ]
# cat_arr = [np.array(Image.open("./cat.1.jpg").resize((100, 100)))for cat in cats[:20]]
# cat_tensor = torch.from_numpy(np.array(img_list))
# print(cat_tensor.size())
# print(cat_tensor.shape)
# print(cat_tensor.dim())
# # 1초 짜리 동영상은 30장의 이미지로 구성되어 있으며
# # 30fps (frames per second)
# # 5(초) x 30 x 100 x 100 x 3 ==> 5 Dimensional tensor
#
#
# # Tensor의 개념, 사용법 등을 확인했음
# # 많이 사용되어지는 연산 종류
# # 뻔하지만 헷갈릴 수 있을 법한거 위주로 정리.
# # 브로드캐스팅 : 마지막 주소 (파이토치 개념) => 자동으로(스스로) 확장해서 계산하는 방법.
# # 행렬 A, B가 있습니다. 두 행렬에 대해 덧셈, 뺄셈을 할 경우
# # 두 행렬의 shape은 같아야 한다? 같아야 한다!
# # 사용 시 주의해서 사용할 것
# # m1 = torch.FloatTensor([ [3, 3] ]) # 1x2 행렬
# # # m2 = torch.FloatTensor([ [2, 2] ]) # 1x2 행렬
# # # m2 = torch.FloatTensor([2]) # 1x1 백터
# # # print(m1.shape, m2.shape) # 브로드캐스팅에 의해서 [[2, 2]]로 자동으로 확장후 연산
# # # print(m1 + m2)
# # #
# # # # m1 = 1x2, m2 = 2x1 ==> 덧셈 수행하면 mxn
# # # # m1+m2 ==> 2x2
# # # m1 = torch.FloatTensor([ [1, 2] ]) # 1x2
# # # m2 = torch.FloatTensor([ [3], [4] ]) # 2x1
# # # print(m1 + m2)
#
# # m1 = torch.FloatTensor([[1, 2]])
# # m2 = torch.FloatTensor([[1, 2]])
# # result = m1 + m2
# # result = torch.add(m1,m2)
# # print(m1+2, m1)
# # _ <== inplace 옵션으로 사용하는 코드가 있음
# # 여러분들은 익숙하기 전까지는 사용x (단, 코드 리뷰할때 보일 수 있으니...참고할 것!)
# # print(m1.add_(2), m1)
#
#
# # 1) Matrix Multiplication
# # 2) Element_Wise곱(Multiplication: 같은 위치에 있는 것끼리 계산)
# m1 = torch.FloatTensor([[1, 2], [3, 4]]) # 2x2 행렬
# # m2 = torch.FloatTensor([[1, 2, 3], [3, 4, 5]]) # 2xn 행렬 (2x3)
# m2 = torch.FloatTensor([[1, 2], [3, 4]])
# # m3 = m1.matmul(m2)
# m4 = m1 * m2    # m1.mul(m2) # Element_Wise 곱
# print(m3)
# print(m4)
# print(m3.shape)
# print(m4.shape)
#
#
# # 다음 시간에 gpu에서 20000x20000 매트릭스 연산하는 것과
# # cpu에서 20000x20000 매트릭스 연산하는 것에 대한 처리 시간 체크
# # 한 번 해보기.

import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
import time
start = time.time()
a = torch.rand(20000, 20000)
b = torch.rand(20000, 20000)
c = a.matmul(b)
end = time.time()
print(f'CPU time: {end - start} seconds')

use_gpu = torch.cuda.is_available()
if use_gpu:
    a = a.cuda()
    b = b.cuda()
    a.matmul(b)
end = time.time()
print(f'GPU time: {end - start} seconds')

import torch
# 평균, max, argmax..
a = torch.FloatTensor([ [1, 4],  [5, 2] ])
# b = torch.FloatTensor([4, 5, 6, 7])
# print(a.mean(dim=0))  # <행=0, 열1> 행기준인지 열기준인지
# print(a.mean(dim=1))
# print(a.sum(dim=0))
# print(a.sum(dim=1))

# print(a.max(dim=0))
# print(a.max(dim=1))
# print(a.argmax())

# shape 변경. 2x2x2 ==> 4x2 ==> 1x2x4
# 2x2x3
t = torch.FloatTensor([
    [
        [1, 2, 3],
        [4, 5, 6]
    ],
    [
        [1, 2, 3],
        [4, 5, 6]
    ]
 ])
print(t.shape) # torch.Size([2, 2, 3]) ==> 3d tensor
# 2d tensor ==> [4x3]
# view (reshape: numpy)
t2 = t.view([-1, 3]) # -1의 의미: 알아서 너가 계산해서 잘 정리해줘.
t3 = t.view([-1, 1, 3])
# mx1x3에서 1,3은 반드시 지키고 나머지는 알아서 계산
# 원본은 2x2x3 ==> mx1x3 ==> "4"x1x3
print(t)
print(t2)
print(t2.shape)

# squeeze: nx1 ==> n으로 만드는거. (하나의 차수를 제거)
t = torch.FloatTensor([ [0], [1], [2], [3]])
print(t)
t_sq = t.squeeze()
print(t_sq)




