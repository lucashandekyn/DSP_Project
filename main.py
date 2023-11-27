from math import *
import scipy.io as sio
from numpy import *


def channel2APDP(frq_char: list) -> list:
    inv_four = fft.ifft(frq_char)
    # power of inverse fourier (for one point
    # ==> needs to be updated for all points)
    sum = 0
    av_power = []
    for j in range(0, 100):
        for i in range(0, 200):
            sum += inv_four[i, j]**2
        av_power.append(sum/(2*200+1))
    return av_power


def APDP2delays():
    print("delay")


def main():
    dataset_1 = sio.loadmat("Dataset_1.mat")
    data = dataset_1["H"]
    punt = data[:, 0, :]
    # print(punt)a
    # APDP = channel2APDP(punt[:200])
    # print(APDP)
    print("Hello World!")


if __name__ == "__main__":
    main()
