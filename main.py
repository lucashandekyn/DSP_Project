from math import *
import scipy.io as sio
from numpy import *
from numpy.fft import *
import matplotlib.pyplot as plt

# Data in vorm van:
# voor transpose => freqtonen (200) | positie(25) | metingen(100)
# na transpose => positie(25) | metingen(100) | freqtonen (200)


def channel2APDP(frq_char: list) -> list:
    data = transpose(frq_char, (1, 2, 0))
    for i in range(0, 100):
        data[:, i] = ifft(data[:, i])
    # power of inverse fourier
    for i in range(0, 100):
        data[:, i] = abs(data[:, i]) ** 2
    # average power delay profile
    av_power = []
    for i in range(0, 100):
        av_power.append(mean(data[:, i]))
    return av_power


def plot_APDP(APDP: list):
    plt.plot(APDP)
    plt.xlabel("delay")
    plt.ylabel("power")
    plt.show()


def APDP2delays():
    print("delay")


def main():
    dataset_1 = sio.loadmat("Dataset_1.mat")
    data = dataset_1["H"]
    APDP = channel2APDP(data)
    print(APDP)
    plot_APDP(APDP)


if __name__ == "__main__":
    main()
