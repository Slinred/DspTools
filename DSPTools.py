import math
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

    def plotFreqz(fs, fcut, filter,freqs, fftResult):
        f, axs = plt.subplots(4,1)
        f.suptitle(f"Filter response with fcut={fcut} @ fs={fs}")
        #fTests = [round((fs/2)/50), round((fs/2)/10), round((fs/2)/5), round((fs/2)/2), round((fs/2)/1.5)]
        fTests = [round((fs/2)/50), round((fs/2)/5), round((fs/2)/2), round((fs/2)/1.5)]
        numSamples=len(freqs)
    
        xTest = np.arange(start=1,stop=1000)
        yTest = 0
        for fTest in fTests:
            yTest += np.sin(2*np.pi*xTest*fTest/fs)
        
        yTestNormalized = (yTest / sum(yTest)) / 20
        yTestFft = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(yTestNormalized,numSamples))))

        filteredTestSig = filter(yTestNormalized)
        filteredTestSigFft = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(filteredTestSig, numSamples))))

        axs[0].plot(freqs,fftResult, 'r-', label="Filter response")
        axs[0].plot(freqs, yTestFft, label=f'Test signal containing {fTests} Hz')
        axs[0].set_title("Frequency domain input")
        axs[0].grid(True)
        axs[0].set_xlim(0, fs/2)
        axs[0].legend(loc=1)

        axs[1].plot(freqs,fftResult, 'r-', label="Filter response")
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

class BiquadFilterFactory:
    # pretend enumeration
    LOWPASS, HIGHPASS, BANDPASS, PEAK, NOTCH, LOWSHELF, HIGHSHELF = range(7)
    
    def calcCoefficients(type, fs, fcut, Q, dbGain = 0):
        types = {
            BiquadFilterFactory.LOWPASS : BiquadFilterFactory.calcLowpass,
            BiquadFilterFactory.HIGHPASS : BiquadFilterFactory.calcHighpass,
            BiquadFilterFactory.BANDPASS : BiquadFilterFactory.calcBandpass,
            BiquadFilterFactory.PEAK : BiquadFilterFactory.calcPeak,
            BiquadFilterFactory.NOTCH : BiquadFilterFactory.calcNotch,
            BiquadFilterFactory.LOWSHELF : BiquadFilterFactory.calcLowshelf,
            BiquadFilterFactory.HIGHSHELF : BiquadFilterFactory.calcHighshelf
        }
        assert type in types
        (A, omega, sn, cs, alpha, beta) = BiquadFilterFactory.calcConstants(fcut, fs, Q, dbGain)
        b0 = b1 = b2 = 0
        a0 = a1 = a2 = 0

        (b0, b1, b2, a0, a1, a2) = types[type](A, omega, sn, cs, alpha, beta)

        b0 /= a0
        b1 /= a0
        b2 /= a0
        a1 /= a0
        a2 /= a0

        #if type == BiquadFilterFactory.LOWPASS:
        #    (b0, b1, b2, a0, a1, a2) = BiquadFilterFactory.calcLowpass(A, omega, sn, cs, alpha, beta)
        #elif type == BiquadFilterFactory.HIGHPASS:
        #    (b0, b1, b2, a0, a1, a2) = BiquadFilterFactory.calcHighpass(A, omega, sn, cs, alpha, beta)
        #elif type == BiquadFilterFactory.BANDPASS:
        #    (b0, b1, b2, a0, a1, a2) = BiquadFilterFactory.calcBandpass(A, omega, sn, cs, alpha, beta)
        #elif type == BiquadFilterFactory.PEAK:
        #    (b0, b1, b2, a0, a1, a2) = BiquadFilterFactory.calcPeak(A, omega, sn, cs, alpha, beta)
        #elif type == BiquadFilterFactory.NOTCH:
        #    (b0, b1, b2, a0, a1, a2) = BiquadFilterFactory.calcNotch(A, omega, sn, cs, alpha, beta)
        #elif type == BiquadFilterFactory.LOWSHELF:
        #    (b0, b1, b2, a0, a1, a2) = BiquadFilterFactory.calcLowshelf(A, omega, sn, cs, alpha, beta)
        #elif type == BiquadFilterFactory.HIGHSHELF:
        #    (b0, b1, b2, a0, a1, a2) = BiquadFilterFactory.calcHighshelf(A, omega, sn, cs, alpha, beta)

        return BiquadFilter.Coefficients(a1, a2, b0, b1, b2, 1)

    def calcConstants(freq:float, fs:float, Q:float, dbGain=0):
        dbGain = float(dbGain)
        # only used for peaking and shelving filter types
        A = math.pow(10, dbGain / 40)
        omega = 2 * math.pi * freq / fs
        sn = math.sin(omega)
        cs = math.cos(omega)
        alpha = sn / (2*Q)
        beta = math.sqrt(A + A)

        return (A, omega, sn, cs, alpha, beta)

    def calcLowpass(A:float, omega:float, sn:float, cs:float, alpha:float, beta:float):
        b0 = (1 - cs) /2
        b1 = 1 - cs
        b2 = (1 - cs) /2
        a0 = 1 + alpha
        a1 = -2 * cs
        a2 = 1 - alpha

        return (b0, b1, b2, a0, a1, a2)

    def calcHighpass(A, omega, sn, cs, alpha, beta):
        b0 = (1 + cs) /2
        b1 = -(1 + cs)
        b2 = (1 + cs) /2
        a0 = 1 + alpha
        a1 = -2 * cs
        a2 = 1 - alpha

        return (b0, b1, b2, a0, a1, a2)

    def calcBandpass(A, omega, sn, cs, alpha, beta):
        b0 = alpha
        b1 = 0
        b2 = -alpha
        a0 = 1 + alpha
        a1 = -2 * cs
        a2 = 1 - alpha

        return (b0, b1, b2, a0, a1, a2)

    def calcNotch(A, omega, sn, cs, alpha, beta):
        b0 = 1
        b1 = -2 * cs
        b2 = 1
        a0 = 1 + alpha
        a1 = -2 * cs
        a2 = 1 - alpha

        return (b0, b1, b2, a0, a1, a2)

    def calcPeak(A, omega, sn, cs, alpha, beta):
        b0 = 1 + (alpha * A)
        b1 = -2 * cs
        b2 = 1 - (alpha * A)
        a0 = 1 + (alpha /A)
        a1 = -2 * cs
        a2 = 1 - (alpha /A)

        return (b0, b1, b2, a0, a1, a2)

    def calcLowshelf(A, omega, sn, cs, alpha, beta):
        b0 = A * ((A + 1) - (A - 1) * cs + beta * sn)
        b1 = 2 * A * ((A - 1) - (A + 1) * cs)
        b2 = A * ((A + 1) - (A - 1) * cs - beta * sn)
        a0 = (A + 1) + (A - 1) * cs + beta * sn
        a1 = -2 * ((A - 1) + (A + 1) * cs)
        a2 = (A + 1) + (A - 1) * cs - beta * sn

        return (b0, b1, b2, a0, a1, a2)

    def calcHighshelf(A, omega, sn, cs, alpha, beta):
        b0 = A * ((A + 1) + (A - 1) * cs + beta * sn)
        b1 = -2 * A * ((A - 1) + (A + 1) * cs)
        b2 = A * ((A + 1) + (A - 1) * cs - beta * sn)
        a0 = (A + 1) - (A - 1) * cs + beta * sn
        a1 = 2 * ((A - 1) - (A + 1) * cs)
        a2 = (A + 1) - (A - 1) * cs - beta * sn
    
        return (b0, b1, b2, a0, a1, a2)

class BiquadFilter:
    class Coefficients():
        def __init__(self, a1:float, a2:float, b0:float, b1:float, b2:float, postShift, a0:float=1):
            self.aFloat = [a0, a1, a2]
            self.aQ15 = Quantization.quantize(self.aFloat, Quantization.Levels.Q15, postShift)
            self.bFloat = [b0, b1, b2]
            self.bQ15 = Quantization.quantize(self.bFloat, Quantization.Levels.Q15, postShift)
            self.postShift = postShift

        def printCoeffs(self):
            print(  f"Coefficients: (post shift = {self.postShift})")
            for a in range(len(self.aFloat)):
                print(f"  a{a}:""\n"f"    {self.aFloat[a]}""\n"f"    q15 = {self.aQ15[a]}")
            for b in range(len(self.bFloat)):
                print(f"  b{b}:""\n"f"    {self.bFloat[b]}""\n"f"    q15 = {self.bQ15[b]}")

    class History:
        def __init__(self):
            self.X1 = 0
            self.X2 = 0
            self.Y1 = 0
            self.Y2 = 0

    def __init__(self, fs, fcut, coeffs, stages):
        self.fcut = fcut
        self.fs = fs
        self.coeffs = coeffs
        self.stages = stages
        self.history = list()
        for stage in range(0,stages):
            self.history.append(self.History())

    # provide a static result for a given frequency f
    def frequencyResponse(self, f):
        phi = (math.sin(math.pi * f * 2/(2*self.fs)))**2
        r =((self.coeffs.bFloat[0]+self.coeffs.bFloat[1]+self.coeffs.bFloat[2])**2 - \
        4*(self.coeffs.bFloat[0]*self.coeffs.bFloat[1] + 4*self.coeffs.bFloat[0]*self.coeffs.bFloat[2] + \
        self.coeffs.bFloat[1]*self.coeffs.bFloat[2])*phi + 16*self.coeffs.bFloat[0]*self.coeffs.bFloat[2]*phi*phi) / \
        ((1+self.coeffs.aFloat[1]+self.coeffs.aFloat[2])**2 - 4*(self.coeffs.aFloat[1] + 4*self.coeffs.aFloat[2] + \
        self.coeffs.aFloat[1]*self.coeffs.aFloat[2])*phi + 16*self.coeffs.aFloat[2]*phi*phi)
        
        if(r < 0):
            r = 0

        return r**(.5)

    # provide a static log result for a given frequency f
    def frequencyResponseLog(self, f):
        try:
            r = 20 * math.log10(self.frequencyResponse(f))
        except:
            r = -200
        return r

    def calcDf1Float(self, input):
        calcCount = 0
        output = np.zeros(len(input))
        for n in range(0,len(output)):
            stageInput = input[n]
            for stage in range(0,self.stages):
                output[n] = (self.coeffs.bFloat[0] * stageInput) + \
                            (self.coeffs.bFloat[1] * self.history[stage].X1) + \
                            (self.coeffs.bFloat[2] * self.history[stage].X2) - \
                            (self.coeffs.aFloat[1] * self.history[stage].Y1) - \
                            (self.coeffs.aFloat[2] * self.history[stage].Y2)
                
                self.history[stage].X2 = self.history[stage].X1
                self.history[stage].X1 = stageInput
                self.history[stage].Y2 = self.history[stage].Y1
                self.history[stage].Y1 = output[n]

                #stageInput = output[n]

                calcCount += 5
        
        print(f"Biquad: calculated {calcCount} muls!")
           
        return output
    
    def calcDf1Q15(self, input):
        raise RuntimeError("Not implemented yet!")

class FirFilterFactory:
    def getLowPassCoefficients(f1,f2, fs, dbAtt=40, showFreqResp=True, N = 0):
        df = f2 - f1
        if N == 0:
            numTaps = round((dbAtt * fs) / (22 * df))
        else:
            numTaps = N
        print("Calculating FIR low pass filter with:\n"
                f"  f(-3dB) = {f1} Hz""\n"
                f"  f(-{dbAtt}db) = {f2} Hz""\n"
                f"  N = {numTaps}")
        fcut = f1 / (fs/2)
        b = signal.firwin(numTaps, fcut)

        if showFreqResp==True:
            FirFilterFactory.plotFreqz(b, fs, f1)
        return b
    
    def plotFreqz(b, fs, fcut, a=1):
        f, axs = plt.subplots(4,1)
        f.suptitle(f"Filter response with fcut={fcut} @ fs={fs}")
        #fTests = [round((fs/2)/50), round((fs/2)/10), round((fs/2)/5), round((fs/2)/2), round((fs/2)/1.5)]
        fTests = [round((fs/2)/50), round((fs/2)/5), round((fs/2)/2), round((fs/2)/1.5)]
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
class FirFilter:
    def __init__(self, coeffs:list, postShift=0, stages=1):
        self.coeffs = coeffs
        self.coeffsQ15 = Quantization.quantize(coeffs, Quantization.Levels.Q15, postShift)
        self.stages = stages
        self.history = list()
        for stage in range(0,self.stages):
            self.history.append(np.zeros(len(self.coeffs)))
    
    def printCoeffsCArray(self, reverse=False):
        coeffs = list(self.coeffsQ15.copy())
        reversed = ""
        if reverse == True:
            coeffs.reverse()
            reversed = "_reversed"
        print(f"int16_t firCoeffs{reversed}[{len(self.coeffsQ15)}] = ""\n{")
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
                self.history[stage][0] = input[n]
                output[n] = 0
                for c in range(0, len(self.coeffs)):
                    if(self.history[stage][c] != 0):
                        output[n] = output[n] + (self.coeffs[c] * self.history[stage][c])
                        calcCount +=1
                    else:
                        skipCount += 1
                #self.history[stage][0] = input[n]
                if n >= 1:
                    for h in range(len(self.history[stage])-1, 0, -1):
                        self.history[stage][h] = self.history[stage][h-1]
        
        print(f"Skipped {skipCount} muls and calculated {calcCount} muls! --> calcCount/skipCount = {calcCount/skipCount}")

        return output

