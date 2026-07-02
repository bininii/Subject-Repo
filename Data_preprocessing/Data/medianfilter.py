# median filter / selected sort algorithm 구현해보기.
def select_sort(in_data):
    pass
def median_filter(in_img):
    select_sort(in_img[0:3, 0:3])
    pass
import cv2
noise_img = cv2.imread('salt-pepper-noise.jpg')
print(type(noise_img), noise_img.shape)
input_img = cv2.cvtColor(noise_img, cv2.COLOR_BGR2GRAY)
filter_img = median_filter(input_img)
print(input_img.shape)
cv2.imshow("noise_img", noise_img)
cv2.waitKey()


# selected sort algorithm(선택정렬) 구현
def select_sort(in_data):
    n = len(in_data)
    for i in range(n):
        # i부터 마지막까지 중 가장 작은값
        min_idx = i
        for j in range(i+1, n):
            if



        # 제일 작은 값 위치
        for j in range(0, n-i):
            if in_data[j] < in_data[min_idx]:
                min_idx = j

        # 제일 작은값을 현재 위치와 바꿈
        in_data[i], in_data[min_idx] = in_data[min_idx], in_data[i]
        return in_data

# # median filter(중앙값) 구현
# def median_filter(in_img):
#
#
# # 결과 출력
# filter_img = median_filter(input_img)
# cv2.imshow
# cv2.waitKey()
# cv2.