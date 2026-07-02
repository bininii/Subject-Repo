### 1.
def impl_pop(input_list, index=None):
    # 마지막 요소 제거
    if index is None:
        if not input_list:
            return -1
        return input_list.pop()

    # 인덱스값 => 넘는 값 들어올때
    if index < 0 or index >= len(input_list):
        return -1

    # 인덱스 요소 반환
    return input_list.pop(index)

input_list = [1, 2, 3, 5, 6, 9]

print(impl_pop(input_list),",",input_list)
print(impl_pop(input_list, 3),",",input_list)