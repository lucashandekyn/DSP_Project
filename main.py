from math import *
import scipy.io as sio
from numpy import *

def channel2APDP(frequentiekarateristiek):
    inv_four = fft.ifft(frequentiekarateristiek)
    return inv_four


def calculate_delays():
    print("delay")


def main():
    dataset_1 = sio.loadmat("Dataset_1.mat")
    data = dataset_1["H"]
    punt = data[:,:,1]
    punt = [rij[0] for rij in punt]
    print(punt)
    print("enterezfndxnvnqsdlnvklnqdflknvklqdfnklvn")
    APDP = channel2APDP(punt[:200])
    print(APDP)
    print("Hello World!")





if __name__ == "__main__":
    main()