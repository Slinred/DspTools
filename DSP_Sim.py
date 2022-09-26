from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import re

from DSPTools import DspTools, FirFilterFactory, TimeSignal, BiquadFilter, FirFilter



def loadCoeffsFromFile(File,sep):
    coeffs = list()
    with open(File, "r") as f:
        while True:
            line = f.readline()
            if line != "":
                if not(line.startswith("#")):
                    matches = re.findall(r"(-*[0-9.]+e*-*[0-9]+)", line)#match = re.findall(r"(-{0,1}[0-9.]+)e*(-*[0-9]+)*", line)
                    for m in matches:
                        coeffs.append(float(m))
            else:
                break
    
    return coeffs

def plotSignalOnAxis(axs, sig:TimeSignal, grid=True, format='-*'):
    if(sig != None):
        axs.plot(sig.x, sig.y, format)
        axs.set_title(sig.name)
        axs.grid(grid)
    return axs

def plotSignals(Signals):
    if len(Signals) > 0:
        shape = np.shape(Signals)
        rows = shape[0]
        if len(shape) == 1:
            cols = 1
        else:
            cols = shape[1]

        f, axs = plt.subplots(rows, cols)

        for r in range(0,rows):
            if cols == 1:
                axs[r] = plotSignalOnAxis(axs[r], Signals[r])
            else:
                for c in range(0,cols):
                    axs[r][c] = plotSignalOnAxis(axs[r][c], Signals[r][c])
        
        # set the spacing between subplots
        plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.4)
        plt.draw()
        plt.pause(0.1)

def main():
    ################################################################################
    # Create sine wave table
    #
    fTable = 18.75
    numSamplesTable = 4096
    numPeriodsTable = 3
    fsTable = (fTable * numSamplesTable) / numPeriodsTable
    sigTable = DspTools.createSineSignal(f"Sine table with {numSamplesTable} samples, f={fTable} Hz", fTable, fsTable, numPeriodsTable)

    ################################################################################

    ################################################################################
    # Sample a sine wave from the table
    #
    fs = 6235
    fSig = 600
    fSigPeriods = 10

    sig = DspTools.sampleSignalFromTable(f"Sampled signal with f = {fSig} Hz @ fs = {fs}", sigTable, fSig, fs, fSigPeriods)

    ################################################################################

    ################################################################################
    # Oversample the signal
    #
    factorOvs = 8
    
    #################
    # Using linear interpolation
    sigOvsLip = DspTools.oversampling(sig, factorOvs, DspTools.OvsTypes.LINEAR_IP)
    
    #################################
    # Using zero padding
    sigOvsZeroPad = DspTools.oversampling(sig, factorOvs, DspTools.OvsTypes.ZERO_PADDING)
    ################################################################################

    ################################################################################
    # Filter the oversampled signal with linar interpoltion with a biquad lpf
    #
    bqCoeffsLpf3kHz = BiquadFilter.Coefficients(-1.4542818459920401, 0.5740622654565052, 0.029945104866116262, 0.059890209732232524, 0.029945104866116262, 1)
    bqCoeffsLpf3kHz.printCoeffs()
    stages = 5
    bqLpf = BiquadFilter(bqCoeffsLpf3kHz, stages)
    sigBqFilteredFloatLip = TimeSignal(f"Filtered ovs lip signal with float precision at fcut=3117Hz and 5 stages",
                                sigOvsLip.x, bqLpf.calcDf1Float(sigOvsLip.y), sigOvsLip.f, sigOvsLip.fs, sigOvsLip.periods)
    
    ################################################################################

    ################################################################################
    # Filter the oversampled signal with zero padding through a poly filter
    #
    firCoeffs = loadCoeffsFromFile("firCoeffsLpf.txt"," ")
    #firCoeffs = FirFilterFactory.getLowPassCoefficients(2000, 3117, 49880)
    firFilter = FirFilter(firCoeffs, 0)
    firFilter.printCoeffsCArray(True)
    sigBqFilteredFloatZp = TimeSignal(f"Filtered ovs zero padding signal with float precision",
                                sigOvsZeroPad.x, firFilter.calcFloat(sigOvsZeroPad.y), sigOvsZeroPad.f, sigOvsZeroPad.fs, sigOvsZeroPad.periods)
    #sigBqFilteredFloatZp.y = sigBqFilteredFloatZp.y * 8
    ################################################################################

    ################################################################################
    # Plot the signals
    #
    plotSignals([sigTable, sig, sigOvsLip, sigOvsZeroPad])
    plotSignals([[sigOvsLip, sigOvsZeroPad],[sigBqFilteredFloatLip, sigBqFilteredFloatZp]])
    
    dPx = np.arange(start=0, step=round(sig.fs/sigTable.f), stop = len(sigTable.y))
    dPy = list()
    for x in dPx:
        dPy.append(sigTable.y[int(x)])

    sx600 = np.arange(start=0, step=round(6235/600), stop=300)
    sy600 = list()
    for i in range(len(sx600)):
        sy600.append(dPy[np.mod(i, len(dPy))])


    f, ax = plt.subplots(2,1)
    ax[0].plot(range(len(sigTable.y)), sigTable.y, 'b-*')
    ax[0].plot(dPx, dPy, 'rX')
    ax[1].plot(sx600, sy600, '-*')


    plt.draw()
    plt.pause(0.1)

    input("Press enter to exit")
    plt.close('all')

if __name__ == "__main__":
    main()