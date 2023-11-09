from math import *
import scipy.io as sio
from numpy import *


def channel2APDP(frq_char: list) -> float:
    inv_four = fft.ifft(frq_char)
    # power of inverse fourier (for one point
    # ==> needs to be updated for all points)
    sum = 0
    av_power = 0
    for j in range(0, 100):
        for i in range(0, 200):
            sum += inv_four[i, j]**2
        av_power += sum/(2*200+1)
    av_power /= 100
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
