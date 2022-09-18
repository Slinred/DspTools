from cProfile import label
from enum import Enum
import numpy as np
from pylab import *
import scipy.signal as signal


class TimeSignal():
    def __init__(self, name, x, y, f, fs, periods):
        self.name = name
        self.x = x
        self.y = y
        self.f = f
        self.fs = fs
        self.periods = periods

class DspTools:
    class OvsTypes(Enum):
        LINEAR_IP = 0
        ZERO_PADDING = 1

    def sampleSignalFromTable(name, table:TimeSignal, fSig, fs, periods=1):
        deltaPhi = round((fs/table.f))
        sigPeriod = round((fs / fSig) +0.5)
        n = sigPeriod * periods
        x = np.arange(start=0, step=1,stop=n) * (1/fs)
        y = np.zeros(n)
        #y[0] = table.y[0]
        for i in range(n):
            y[i] = table.y[np.mod((i*deltaPhi), len(table.y))]#round(len(table.y)/table.periods))]

        return TimeSignal(name, x, y, fSig, fs, periods)

    def createSineSignal(name, f, fs, periods):
        x = np.arange(start=0, step=1/fs, stop=periods/f)
        y = 0.99 * np.sin(2*np.pi*x * f)

        return TimeSignal(name, x,y,f,fs,periods)

    def linearInterpolation(lastSample, nextSample, factor):
        out = (np.arange(start=1/factor, step=1/factor, stop=1) * \
                (nextSample-lastSample)) + lastSample
        return out


    def oversampling(sig:TimeSignal, factor, ovsType:OvsTypes):
        fsOvs = sig.fs * 8
        numSamplesOvs = 8 * len(sig.y)
        xOvs = np.arange(start=0, step=1/fsOvs, stop=numSamplesOvs/fsOvs)
        yOvs = np.zeros(len(sig.y) * factor)
        for i in range(0,len(sig.y)):
            yOvs[i*factor] = sig.y[i]
            startIdx = ((i*factor)+1)
            stopIdx = (((i+1)*factor))
            if ovsType == DspTools.OvsTypes.LINEAR_IP:
                yOvs[startIdx:stopIdx] = \
                    (np.arange(start=1/factor, step=1/factor, stop=1) * \
                    (sig.y[np.mod(i+1, len(sig.y))]-sig.y[i])) + sig.y[i]
                yOvs[startIdx:stopIdx] = DspTools.linearInterpolation(sig.y[i], sig.y[np.mod(i+1, len(sig.y))], factor)
            
            elif ovsType == DspTools.OvsTypes.ZERO_PADDING:
                yOvs[startIdx:stopIdx] = np.zeros(factor-1)

        return TimeSignal(f"{factor} times oversampled signal with {ovsType.name} (f={sig.f} Hz @ fs={fsOvs} Hz)",
                        xOvs, yOvs, sig.f, fsOvs, sig.periods)

class Quantization:
    class Levels(Enum):
        Q15 = 15
        Q31 = 31
    
    def quantize(floatInput:float, q:Levels, postShift):
        quantizedInput = np.zeros(len(floatInput))
        for i in range(len(floatInput)):
            input = floatInput[i] * pow(2,-postShift) # scale the coeff if needed
            if input >= -1 and input <= 1:
                quantizedInput[i] = input * pow(2,q.value)
            else:
                raise ValueError(f"Input {input} not in the range +1..-1!" + "\n\t"
                    "If you need larger input values, scale the input down and use the postShift bits for coefficients in the filter function!")

        if q == Quantization.Levels.Q15:
            return np.int16(quantizedInput)
        elif q == Quantization.Levels.Q31:
            return np.int32(quantizedInput)

class BiquadFilter:
    class BQ_Coeffs():
        def __init__(self, a1:float, a2:float, b0:float, b1:float, b2:float, postShift):
            self.aFloat = (a1, a2)
            self.aQ15 = Quantization.quantize(self.aFloat, Quantization.Levels.Q15, postShift)
            self.bFloat = (b0, b1, b2)
            self.bQ15 = Quantization.quantize(self.bFloat, Quantization.Levels.Q15, postShift)
            self.postShift = postShift
        def printCoeffs(self):
            print(  f"Coefficients: (post shift = {self.postShift})")
            for a in range(len(self.aFloat)):
                print(f"  a{a+1}:""\n"f"    {self.aFloat[a]}""\n"f"    q15 = {self.aQ15[a]}")
            for b in range(len(self.bFloat)):
                print(f"  b{b}:""\n"f"    {self.bFloat[b]}""\n"f"    q15 = {self.bQ15[b]}")

    class BQ_History:
        def __init__(self):
            self.X1 = 0
            self.X2 = 0
            self.Y1 = 0
            self.Y2 = 0

    def __init__(self, coeffs, stages):
        self.coeffs = coeffs
        self.stages = stages
        self.history = list()
        for stage in range(0,stages):
            self.history.append(self.BQ_History())

    def calcDf1Float(self, input):
        calcCount = 0
        output = np.zeros(len(input))
        for n in range(0,len(output)):
            stageInput = input[n]
            for stage in range(0,self.stages):
                output[n] = (self.coeffs.bFloat[0] * stageInput) + \
                            (self.coeffs.bFloat[1] * self.history[stage].X1) + \
                            (self.coeffs.bFloat[2] * self.history[stage].X2) - \
                            (self.coeffs.aFloat[0] * self.history[stage].Y1) - \
                            (self.coeffs.aFloat[1] * self.history[stage].Y2)
                
                self.history[stage].X1 = stageInput
                self.history[stage].X2 = self.history[stage].X1
                self.history[stage].Y1 = output[n]
                self.history[stage].Y2 = self.history[stage].Y1

                #stageInput = output[n]

                calcCount += 5
        
        print(f"Biquad: calculated {calcCount} muls!")
           
        return output
    
    def calcDf1Q15(self, input):
        print("Not implemented yet!")

class FirFilterFactory:
    def getLowPassCoefficients(f1,f2, fs, dbAtt=40, showFreqResp=True, N = 0):
        df = f2 - f1
        if N == 0:
            nCoeffs = round((dbAtt * fs) / (22 * df))
        else:
            nCoeffs = N
        print("Calculating FIR low pass filter with:\n"
                f"  f(-3dB) = {f1} Hz""\n"
                f"  f(-{dbAtt}db) = {f2} Hz""\n"
                f"  N = {nCoeffs}")
        fcut = f1 / (fs/2)
        b = signal.firwin(numtaps=nCoeffs, fs=fs, cutoff=fcut)

        if showFreqResp==True:
            FirFilterFactory.plotFreqz(b, fs, f1)
            #FirFilterFactory.plotImpz(ax[2:len(ax)], b)
        return b

    def plotFreqz(b, fs, fcut, a=1):
        f, axs = plt.subplots(4,1)
        f.suptitle(f"Filter response with fcut={fcut} @ fs={fs}")
        fTests = [round((fs/2)/50), round((fs/2)/10), round((fs/2)/5), round((fs/2)/2), round((fs/2)/1.5)]
        
        numSamples=4096

        xTest = np.arange(start=1,stop=1000)
        yTest = 0
        for fTest in fTests:
            yTest += np.sin(2*np.pi*xTest*fTest/fs)
        
        yTestNormalized = (yTest / sum(yTest)) / 20
        yTestFft = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(yTestNormalized,numSamples))))

        freqs = np.arange(start=-0.5, step=1/numSamples, stop=0.5)*fs
        fftRes = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(b,numSamples))))
        
        firFilter = FirFilter(b)

        filteredTestSig = firFilter.calcFloat(yTestNormalized)
        filteredTestSigFft = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(filteredTestSig, numSamples))))

        axs[0].plot(freqs,fftRes, 'r-', label="Filter response")
        axs[0].plot(freqs, yTestFft, label=f'Test signal containing {fTests} Hz')
        axs[0].set_title("Frequency domain input")
        axs[0].grid(True)
        axs[0].set_xlim(0, fs/2)
        axs[0].legend(loc=1)

        axs[1].plot(freqs,fftRes, 'r-', label="Filter response")
        axs[1].plot(freqs, filteredTestSigFft, label=f'Filtered test signal')
        axs[1].set_title("Frequency domain response")
        axs[1].grid(True)
        axs[1].set_xlim(0, fs/2)
        axs[1].set_ylim(-100, 60)
        axs[1].legend(loc=1)

        axs[2].plot(xTest, yTest, label=f'Test signal containing {fTests} Hz')
        axs[2].set_title("Filter input")
        axs[2].grid(True)

        axs[3].plot(xTest, filteredTestSig, label=f'Filtered test signal')
        axs[3].set_title("Filter output")
        axs[3].grid(True)


        f.subplots_adjust(hspace=0.5)

        plt.draw()
        plt.pause(0.1)


    #Plot frequency and phase response
    def NplotFreqz(axs, b, a=1):
        w,h = signal.freqz(b,a)
        h_dB = 20 * log10 (abs(h))
        axs[0].plot(w/max(w),h_dB)
        axs[0].set_ylim(-150, 5)
        axs[0].set_ylabel('Magnitude (db)')
        axs[0].set_xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
        axs[0].set_title(r'Frequency response')
        h_Phase = unwrap(arctan2(imag(h),real(h)))
        axs[1].plot(w/max(w),h_Phase)
        axs[1].set_ylabel('Phase (radians)')
        axs[1].set_xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
        axs[1].set_title(r'Phase response')

    #Plot step and impulse response
    def NplotImpz(axs, b, a=1):
        l = len(b)
        impulse = repeat(0.,l); impulse[0] =1.
        x = arange(0,l)
        response = signal.lfilter(b,a,impulse)
        axs[0].stem(x, response)
        axs[0].set_ylabel('Amplitude')
        axs[0].set_xlabel(r'n (samples)')
        axs[0].set_title(r'Impulse response')

        step = cumsum(response)
        axs[1].stem(x, step)
        axs[1].set_ylabel('Amplitude')
        axs[1].set_xlabel(r'n (samples)')
        axs[1].set_title(r'Step response')

class FirFilter:
    def __init__(self, coeffs:list, postShift=0, stages=1):
        self.coeffs = coeffs
        self.coeffsQ15 = Quantization.quantize(coeffs, Quantization.Levels.Q15, postShift)
        self.stages = stages
        self.history = list()
        for stage in range(0,self.stages):
            self.history.append(np.zeros(len(self.coeffs)))
    
    def printCoeffsCArray(self, reverse=False):
        print(f"int16_t firCoeffs[{len(self.coeffsQ15)}] = ""\n{")
        coeffs = list(self.coeffsQ15.copy())
        if reverse == True:
            coeffs.reverse()
        for c in range(0, len(coeffs), 8):
            line = "  "
            for cc in range(8):
                if (c+cc) < len(coeffs): 
                    line += f"{coeffs[c+cc]}, "
                else:
                    break
            print(line)
        print("};")

    def calcFloat(self, input):
        skipCount = 0
        calcCount = 0
        output = np.zeros(len(input))
        for n in range(0,len(input)):
            for stage in range(0, self.stages):
                if(input[n] != 0):
                    output[n] = self.coeffs[0] * input[n]
                    calcCount +=1
                else:
                    skipCount += 1
                for c in range(1, len(self.coeffs)):
                    if(self.history[stage][c] != 0):
                        output[n] = output[n] + (self.coeffs[c] * self.history[stage][c])
                        calcCount +=1
                    else:
                        skipCount += 1
                #self.history[stage][0] = input[n]
                if n >= 1:
                    for h in range(len(self.history[stage])-1, 0, -1):
                        self.history[stage][h] = self.history[stage][h-1]
                self.history[stage][0] = input[n]
        
        print(f"Skipped {skipCount} muls and calculated {calcCount} muls! --> calcCount/skipCount = {calcCount/skipCount}")

        return output

