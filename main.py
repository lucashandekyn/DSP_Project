from math import *
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

# Data in vorm van:
# voor transpose => freqtonen (200) | positie(25) | metingen(100)
# na transpose => positie(25) | metingen(100) | freqtonen (200)


def channel2APDP(dataset: list, pos: int) -> list:
    dataset = np.transpose(dataset, (1, 2, 0))
    dataset = np.reshape(dataset, (25, 100, 200))

    apdp = 0

    """     fs=1e9
    freqkar = dataset[pos-1][0][:]
    impulsrespons = fftpack.ifft(freqkar)
    pdp = np.abs(impulsrespons)**2
    tijdstappen = np.arange(0, len(pdp)) / fs

    #plt.plot(impulsrespons)
    plt.plot(tijdstappen, pdp)
    plt.show() """


    for i in range(100):
        freqkar = dataset[pos-1][i][:]
        impulsrespons = fftpack.ifft(freqkar)
        pdp = np.abs(impulsrespons)**2
        #vermogen = sum(abs(x)**2 for x in impulsrespons) / len(impulsrespons)
        apdp += pdp
    apdp = apdp / 100



    #print(apdp)
    return apdp


def APDP2delays(apdp):
    lokale_maxima_index = [i for i in range(1, len(apdp)-1) if apdp[i] > apdp[i-1] and apdp[i] > apdp[i+1]]

    # Sorteer de lokale maxima op basis van het vermogen in aflopende volgorde
    gesorteerde_maxima = sorted(lokale_maxima_index, key=lambda i: apdp[i], reverse=True)

    # Selecteer de twee grootste lokale maxima (als ze bestaan)
    grootste_maxima = gesorteerde_maxima[:2] if len(gesorteerde_maxima) >= 2 else gesorteerde_maxima
    # Zet getal om naar delay
    grootste_maxima = [1/(1e9 + x * 10e6) for x in grootste_maxima]
    print(grootste_maxima)
    return grootste_maxima


def calculate_delays(dataset: list) -> list: 
    maxi = []
    for i in range(len(dataset[0][:][:])):
        maxi.append(APDP2delays(channel2APDP(dataset, i+1)))
    print(len(maxi))
    return maxi

def calculate_location(delay_list: list):

    pass


def main():
    dataset_1 = sio.loadmat("Dataset_1.mat")
    dataset_1 = dataset_1["H"]
    calculate_delays(dataset_1)

    #print(APDP)
    #plot_APDP(APDP)
    


    ## Het echte traject in een plot ##
    t = np.linspace(0, 25, 100)
    x = 2 + t/2
    y = t**2/32 - t/2 + 6

    plt.plot(x,y, label='echt traject')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
