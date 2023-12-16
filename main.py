from math import *
import scipy.io as sio
import scipy.constants as const
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy import signal


def channel2APDP(dataset: list, pos: int, N: int, windowb: bool) -> list:
    ## De dataset wordt omgevormd naar een simpelere bewerkbare dataset
    #   voor transpose => freqtonen (200) | positie(25) | metingen(100)
    #   na transpose => positie(25) | metingen(100) | freqtonen (200)
    dataset = np.transpose(dataset, (1, 2, 0))
    PDP = []

    ## Filter toepassen ##
    if windowb:
        window = signal.windows.gaussian(N,26)
        dataset = [[meting*window for meting in pos] for pos in dataset]
    
    ## Voor elk van de 100 metingen wordt het vermogen van de inverse fourier bepaalt om daarna het gemiddelde te nemen
    for i in range(len(dataset[pos])):
        freqkar = dataset[pos][i][:]
        vermogen = (abs(fftpack.ifft(freqkar)))
        PDP.append(vermogen)
    APDP = np.mean(PDP, axis=0)

    #plot_APDP(vermogen)
    return APDP


def plot_APDP(vermogen: list):
    plt.plot(vermogen)
    plt.xlabel("delay")
    plt.ylabel("power")   
    plt.show()


def APDP2delays(vermogen: list, dT: int) -> list:
    ## Identificeer pieken waar de lokale maxima kunnen liggen
    piek_indexen, NA = signal.find_peaks(vermogen)

    ## Sorteer de pieken op basis van hoogte (vermogen)
    gesorteerde_pieken = sorted(piek_indexen, key= lambda i: vermogen[i], reverse=True)

    ## Pak de 2 grootste pieken als τ0 en τ1
    twee_grootste_pieken = gesorteerde_pieken[:2] if len(gesorteerde_pieken) >= 2 else gesorteerde_pieken
    
    ## Zet de gekregen index om naar de werkelijke delay
    twee_grootste_pieken = [peak * dT for peak in twee_grootste_pieken]
    #print(twee_grootste_pieken)
    
    return twee_grootste_pieken


def calculate_delays(dataset: list, fs: int, window: bool) -> list:
    delays = []
    N = len(dataset[:][:][:])       # Aantal sampels in de dataset    
    dT = 1/(N * fs)                 # Afstand tussen 2 tijdssamples

    ## Voor alle 25 punten wordt eerst de APDP berekend en daaruit worden de delays berekend          
    for i in range(len(dataset[0][:][:])):
        vermogen = channel2APDP(dataset, i, N, window)
        delays.append(APDP2delays(vermogen, dT))
    return delays


def calculate_location(delays: list) -> list:
    # t0: tau 0 ==> rijstijd direct pad
    # t1: tau 1 ==> rijstijd gereflecteerd pad
    # [xb0,yb0] = [0,1] (coordinaten basisstation)
    # gereflecteerd: [xb1,yb1] = [0,-1]
    # ==> hetzelfde pad als je alles zou optellen met reflectie
    # afstand == tijd * snelheid
    y0 = 1
    y1 = -1
    d = 2       # Afstand tussen de 2 middelpunten
    locations = []
    locationsx = []
    locationsy = []

    ## Bereken voor elk punt de locatie doormiddel van de rijstijden (delays)
    for delay in delays:
        t0 = delay[0]
        t1 = delay[1]
        r0 = t0 * const.c 
        r1 = t1 * const.c 

        ## We nemen 2 cirkels met als middelpunten het basistation het het greflecteerde basistation
        ## De gezochte locatie is waar deze 2 cirkel elkaar snijden (in het rechtse vlak)
        a = (r0**2 - r1**2 + d**2) / (2 * d)
        if r0 > abs(a):
            #print(r0, r1, a)
            h = sqrt(r0**2 - a**2)
            x = h*(y0 - y1) / d            #x0 + a*(x1 - x0) / d + h*(y1 - y0) / d
            y = y0 + a*(y1 - y0) / d       #y0 + a*(y1 - y0) / d - h*(x1 - x0) / d
            locations.append([x,y])
            locationsx.append(x)
            locationsy.append(y)
        else:
            print(delay)
        
    return locationsx, locationsy


def mediaanfout(locationsx: list, locationsy: list, echt_locationsx: list, echt_locationsy: list) -> list:
    xfout = []
    yfout = []
    for i in range(len(locationsx)):
        xfout.append(abs(echt_locationsx[i] - locationsx[i]))
        yfout.append(abs(echt_locationsy[i] - locationsy[i]))

    return np.median(xfout), np.median(yfout)


def echte_traject():
    x = []
    y = []
    for i in range(25):
        x.append(2 + i/2)
        y.append(i**2/32 - i/2 +6)
    return x,y


def main():
    #### Dataset 1 ####
    dataset_1 = sio.loadmat("Dataset_1.mat")
    dataset_1 = dataset_1["H"]
    fs1 = 10e6  # Om de 10MHz een sample dus fs=10e6
    
    ## Bereken de delays uit de frequentiekarakteristiek zonder window
    delays = calculate_delays(dataset_1, fs1, False)

    ## Bereken de delays uit de frequentiekarakteristiek met window
    delaysW = calculate_delays(dataset_1, fs1, True)
    #print(delays)

    ## Bereken de locatie van de punten uit de delays
    locationsx, locationsy = calculate_location(delaysW)
    print("Locaties voor Dataset 1")
    
    for i in range(len(locationsx)):
        print(f"Punt {i}: ({locationsx[i]}, {locationsy[i]})")
    
    ## Bepaal de mediaan fout op de lokalisatie
    x, y = echte_traject()
    print("Mediaanfout = ", mediaanfout(locationsx, locationsy,x, y))

    ## Plot het gereconstrueerd traject
    plt.plot(locationsx, locationsy, label='gereconstrueerd traject')

    ## Plot het echte traject
    plt.plot(x, y, label='echt traject')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


    #### Dataset 2 ####
    dataset_2 = sio.loadmat("Dataset_2.mat")
    dataset_2 = dataset_2["H"]
    fs2 = 10e6

    ## Bereken de delays uit de frequentiekarakteristiek zonder window
    delays = calculate_delays(dataset_2, fs2, False)
    #print(delays)

    ## Bereken de delays uit de frequentiekarakteristiek met window
    delaysW = calculate_delays(dataset_2, fs2, True)

    ## Bereken de locatie van de punten uit de delays
    locationsx, locationsy = calculate_location(delaysW)
    print("Locaties voor Dataset 2")

    for i in range(len(locationsx)):
        print(f"Punt {i}: ({locationsx[i]}, {locationsy[i]})")
    
    ## Bepaal de mediaan fout op de lokalisatie
    x, y = echte_traject()
    print("Mediaanfout = ", mediaanfout(locationsx, locationsy,x, y))

    ## Plot het gereconstrueerd traject
    plt.plot(locationsx,locationsy, label='gereconstrueerd traject')

    ## Plot het echte traject
    plt.plot(x, y, label='echt traject')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

main()
