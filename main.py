from math import *
import scipy.io as sio
from numpy import *

def channel2APDP(frequentiekarateristiek):
    inv_four = fft.ifft(frequentiekarateristiek)
    


def calculate_delays():
    print("delay")


def main():
    dataset_1 = sio.loadmat("Dataset_1.mat")
    print(dataset_1)
    print("Hello World!")





if __name__ == "__main__":
    main()