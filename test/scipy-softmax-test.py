# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import softmax


def main():
    np.set_printoptions(precision=5)
    x = np.array([[1, 0.5, 0.2, 3],
                  [1,  -1,   7, 3],
                  [2,  12,  13, 3]])
    print('x:shape:{}\n{}'.format(x.shape, x))
    print('#' * 80)

    #
    # row-wise
    #
    softmax_list = list()

    for i in range(x.shape[0]):
        exp_row = [np.exp(c) for c in x[i]]
        exp_sum = sum(exp_row)
        softmax_row = [c / exp_sum for c in exp_row]
        softmax_list.append(softmax_row)
        print('{} / {} / {} --> {}'.format(x[i], exp_row, exp_sum, softmax_row))
        print('+' * 10)
        pass

    softmax_array = np.array(softmax_list)
    print('#1-1:softmax_array:\n{}'.format(softmax_array))
    print('+' * 10)
    print('+' * 10)
    softmax_array = softmax(x, axis=1)
    print('#1-2:softmax_array:\n{}'.format(softmax_array))

    print('#' * 80)

    #
    # column-wise
    #
    softmax_list = list()

    for i in range(x.shape[1]):
        exp_row = [np.exp(c) for c in x[:,i]]
        exp_sum = sum(exp_row)
        softmax_row = [c / exp_sum for c in exp_row]
        softmax_list.append(softmax_row)
        print('{} / {} / {} --> {}'.format(x[:,i], exp_row, exp_sum, softmax_row))
        print('+' * 10)
        pass

    softmax_array = np.array(softmax_list).T
    print('#2-1:softmax_array:\n{}'.format(softmax_array))
    print('+' * 10)
    print('+' * 10)
    softmax_array = softmax(x, axis=0)
    print('#2-2:softmax_array:\n{}'.format(softmax_array))

    print('#' * 80)

    #
    # element-wise
    #
    exp_array = np.exp(x)
    exp_sum = np.sum(exp_array)
    softmax_array = exp_array / exp_sum
    print('#3-1:exp_array:\n{}'.format(exp_array))
    print('exp_sum:{}'.format(exp_sum))
    print('+' * 10)
    print('#3-1:softmax_array:\n{}'.format(softmax_array))

    print('+' * 10)
    print('+' * 10)

    softmax_array = softmax(x) # default axis=None
    print('#3-2:softmax_array:\n{}'.format(softmax_array))

    pass

if __name__ == '__main__':
    main()
    pass
