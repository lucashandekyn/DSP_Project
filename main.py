from math import *
import scipy.io as sio
import scipy.constants as const
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

# Data in vorm van:
# voor transpose => freqtonen (200) | positie(25) | metingen(100)
# na transpose => positie(25) | metingen(100) | freqtonen (200)


def channel2APDP(dataset: list, pos: int) -> list:
    dataset = np.transpose(dataset, (1, 2, 0))
    dataset = np.reshape(dataset, (25, 100, 200))
    PDP = []
    for i in range(len(dataset[pos])-1):
        freqkar = dataset[pos][i][:]
        vermogen = (abs(fftpack.ifft(freqkar)))**2
        PDP.append(vermogen)
    APDP = np.mean(PDP, axis=0)
    print("APDP: ", APDP)
    return APDP


# def plot_APDP(APDP: list):
#     plt.plot(APDP)
#     plt.xlabel("delay")
#     plt.ylabel("power")
#     plt.show()


# geeft 2 grootste maxima uit de APDP
def APDP2delays(apdp):
    lokale_maxima_index = [i for i in range(
        1, len(apdp)-1) if apdp[i] > apdp[i-1] and apdp[i] > apdp[i+1]]

    # Sorteer de lokale maxima op basis van het vermogen in aflopende volgorde
    gesorteerde_maxima = sorted(
        lokale_maxima_index, key=lambda i: apdp[i], reverse=True)

    # Selecteer de twee grootste lokale maxima (als ze bestaan)
    grootste_maxima = gesorteerde_maxima[:2] if len(
        gesorteerde_maxima) >= 2 else gesorteerde_maxima
    # Zet getal om naar delay
    grootste_maxima = [1/(1e9 + x * 10e6) for x in grootste_maxima]
    return grootste_maxima


def calculate_delays(dataset: list) -> list:
    maxi = []
    for i in range(len(dataset[0][:][:])):
        maxi.append(APDP2delays(channel2APDP(dataset, i)))
    return maxi


def calculate_location(delays: list) -> list:
    # t0: tau 0 ==> rijstijd direct pad
    # t1: tau 1 ==> rijstijd gereflecteerd pad
    # [xb0,yb0] = [0,1] (coordinaten basisstation)
    # gereflecteerd: [xb1,yb1] = [0,-1]
    # ==> hetzelfde pad als je alles zou optellen met reflectie
    # afstand == tijd * snelheid
    y0 = 1
    y1 = -1
    locations = []
    for delay in delays:
        t0 = delay[0]
        t1 = delay[1]
        r0 = t0 * const.c
        r1 = t1 * const.c
        x = 0  # todo
        y = 0  # todo
        locations.append([x, y])
    return locations


def main():
    dataset_1 = sio.loadmat("Dataset_1.mat")
    dataset_1 = dataset_1["H"]
    channel2APDP(dataset_1, 0)
    delays = calculate_delays(dataset_1)
    locations = calculate_location(delays)
    print(locations)

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
