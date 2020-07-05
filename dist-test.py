# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import distance

def main():
    a = np.array((1, 2, 3))
    b = np.array((4, 5, 6))

    dist = np.linalg.norm(a -b)
    print('#1:a:{}, b:{} --> dist:{}'.format(a, b, dist))

    #dist = scipy.spatial.distance.euclidean(a, b)
    dist = distance.euclidean(a, b)
    print('#2:a:{}, b:{} --> dist:{}'.format(a, b, dist))
    pass

if __name__ == '__main__':
    main()
    pass
