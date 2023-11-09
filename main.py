from math import *
import scipy.io as sio
from numpy import *


def channel2APDP(frq_char):
    inv_four = fft.ifft(frq_char)
    # power of inverse fourier (for one point
    # ==> needs to be updated for all points)
    sum = 0
    for i in range(0, 200):
        sum += inv_four[i]**2
    power = sum/(2*200+1)
    av_power = 0
    for i in range(0,100):
        
    return av_power


def APDP2delays():
    print("delay")


def main():
    dataset_1 = sio.loadmat("Dataset_1.mat")
    data = dataset_1["H"]
    punt = data[:, :, 1]
    punt = [rij[0] for rij in punt]
    print(punt)
    APDP = channel2APDP(punt[:200])
    print(APDP)
    print("Hello World!")


if __name__ == "__main__":
    main()
