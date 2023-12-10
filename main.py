from math import *
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

# Data in vorm van:
# voor transpose => freqtonen (200) | positie(25) | metingen(100)
# na transpose => positie(25) | metingen(100) | freqtonen (200)


def channel2APDP(dataset: list, pos: int) -> float:
    dataset = np.transpose(dataset, (1, 2, 0))
    dataset = np.reshape(dataset, (25, 100, 200))
    APDP = 0
    freqkar = dataset[pos][0][:]
    impulsrespons = fftpack.ifft(freqkar)
    plt.plot(impulsrespons)
    plt.show()
    for i in range(100):
        freqkar = dataset[pos][i][:]
        impulsrespons = np.real(fftpack.ifft(freqkar))
        vermogen = sum(abs(x)**2 for x in impulsrespons) / len(impulsrespons)
        APDP += vermogen
    APDP = APDP / 100
    print(APDP)
    return APDP


def plot_APDP(APDP: list):
    plt.plot(APDP)
    plt.xlabel("delay")
    plt.ylabel("power")
    plt.show()


# geeft 2 grootste maxima uit de APDPs
def APDP2delays(apdps: list) -> list:
    lokale_maxima_index = [i for i in range(
        1, len(apdps)-1) if apdps[i] > apdps[i-1] and apdps[i] > apdps[i+1]]

    # Sorteer de lokale maxima op basis van het vermogen in aflopende volgorde
    gesorteerde_maxima = sorted(
        lokale_maxima_index, key=lambda i: apdps[i], reverse=True)

    # Selecteer de twee grootste lokale maxima (als ze bestaan)
    grootste_maxima = gesorteerde_maxima[:2] if len(
        gesorteerde_maxima) >= 2 else gesorteerde_maxima
    print(grootste_maxima)
    return grootste_maxima


def calculate_delays(dataset) -> list:
    apdp = []
    maxi = []
    for i in range(25):
        apdp.append(channel2APDP(dataset, i))
        maxi.append(APDP2delays(apdp[i]))
    return maxi


def calculate_location():

    pass


def main():
    dataset_1 = sio.loadmat("Dataset_1.mat")
    dataset_1 = dataset_1["H"]

    delays = calculate_delays(dataset_1)
    print(delays)

    ## Het echte traject in een plot ##
    t = np.linspace(0, 25, 100)
    x = 2 + t/2
    y = t**2/32 - t/2 + 6

    plt.plot(x, y, label='echt traject')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


main()
