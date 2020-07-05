# -*- coding: utf-8 -*-

import numpy as np

def main():
    a = np.array([[ 1,  2,  3], [ 4,  5,  6]])
    b = np.array([[11, 12, 13], [14, 15, 16]])
    print('a:shpae:{}\n{}'.format(a.shape, a))
    print('+' * 10)
    print('b:shpae:{}\n{}'.format(b.shape, b))
    print('+' * 10)

    axis = 1
    c = np.concatenate((a, b), axis=axis)
    print('c:shpae:{}\n{}'.format(c.shape, c))
    print('+' * 10)
    axis = 0
    c = np.concatenate((a, b), axis=axis)
    print('c:shpae:{}\n{}'.format(c.shape, c))
    pass

if __name__ == '__main__':
    main()
    pass
