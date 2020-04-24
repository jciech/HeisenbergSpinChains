import numpy as np
import scipy.sparse

def countBits(x):
    """
    The function counts the number of 1's in the binary representation of a base 10 integer.

    :param x: Base 10 integer
    :return: The number of ones of the integer in binary representation
    """
    # from https://stackoverflow.com/questions/10874012/how-does-this-bit-manipulation-work-in-java/10874449#10874449
    # Used because of the O(log(n)) complexity

    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    x = x + (x >> 8)
    x = x + (x >> 16)
    return x & 0x0000003F


def intToBinary(x, N):
    """
    The function converts a base 10 integer into a binary number of a specified length (padding with 0s if needed)

    :param x: Base 10 integer to be converted to binary
    :param N: Length of binary string required
    :return: Binary number of specified length
    """
    return ('{0:0' + str(N) + 'b}').format(x)


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def normalizeSparse(v):
    norm = scipy.sparse.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm