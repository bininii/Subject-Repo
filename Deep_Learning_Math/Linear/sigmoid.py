import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
    x = np.arange(-5.0, 5.0, 0.1)
    # print(x)
    # w, b 값 변경하면서 해보세요. (for문으로 그려봐도 되고..)
    w = [2.2, 0.8, 1.5]
    b = 2
    linear_val = w[0]*x + b
    y0 = sigmoid(linear_val)
    linear_val = w[1]*x + b
    y1 = sigmoid(linear_val)
    # print(y)
    plt.plot(x, y0, 'g')
    plt.plot(x, y1, 'r--')
    plt.plot([0,0], [1,0], ":")
    plt.title('sigmoid function')
    plt.show()
