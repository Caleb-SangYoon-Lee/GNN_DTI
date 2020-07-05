# -*- coding: utf-8 -*-

def good_enough(guess, x):
    return abs(guess ** 2 - x) < 0.0001

def improve(guess, x):
    return (guess + x / guess) / 2

def sqrt_iter(guess, x):
    if good_enough(guess, x):
        return guess
    else:
        return sqrt_iter(improve(guess, x), x)
    pass

def sqrt(x):
    return sqrt_iter(1.0, x)

def main():
    for x in range(1, 11):
        print('x:{} --> {}'.format(x, sqrt(x)))
        pass
    pass

if __name__ == '__main__':
    main()
    pass
