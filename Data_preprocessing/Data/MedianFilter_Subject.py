import cv2
import numpy as np

# 모든픽셀 하나씩 방문
# 각위치마다 3x3 픽셀 9개 가져옴
# 9개 선택정렬로 정렬
# 정렬된 9개중 중간값(median) 뽑아서 이미지 원래위치에 저장
def select_sort(in_data):
    n = len(in_data)
    for i in range(len(in_data)):
        min_index = i
        for j in range(i+1, len(in_data)):
            if in_data[min_index] > in_data[j]:
                min_index = j
        in_data[i], in_data[min_index] = in_data[min_index], in_data[i]
    return in_data

def median_filter(in_img, pad):
    height, width = in_img.shape[:2]
    # 결과 저장할 빈 이미지
    result_image = np.zeros((height, width), dtype=np.uint8)


    # 테두리 제외한 모든 픽셀 방문
    for y in range(1, height - 1):
        for x in range(1, width - 1):

            # (y,x) 중심으로 3x3 픽셀 9개 모으기
            pixel_list = []
            for ky in range(-1, 2):
                for kx in range(-1, 2):
                    pixel_list.append(in_img[y + ky, x + kx])


            # 선택정렬 => 9개 정렬
            sorted_list = select_sort(pixel_list)

            # 정렬된 9개중 중간값(5번째 값) 뽑아서 이미지 원래위치에 저장
            result_image[y, x] = sorted_list[4]

    return result_image

# 이미지 불러오기
path = './saltpepper_noise.png'
noise_img = cv2.imread(path)
in_img = cv2.cvtColor(noise_img, cv2.COLOR_BGR2GRAY)
filter_img = cv2.medianBlur(noise_img, 3)

filter_img = median_filter(in_img, 3)
cv2.imshow("noise", in_img)
cv2.imshow("filter_img", filter_img)
cv2.waitKey()