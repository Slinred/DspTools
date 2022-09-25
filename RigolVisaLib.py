from enum import Enum
import re
import numpy as np
import pyvisa as pv

class ExtendedEnum(Enum):

    @classmethod
    def list(cls):
        return list(map(lambda c: c, cls))
    
    @classmethod
    def getEnumFromStr(cls, strVal):
        vals = cls.list()
        for v in vals:
            if v.value == strVal:
                return v
        return None

class RigolBwLimit(ExtendedEnum):
    OFF = "OFF"
    ON_20MHz = "20M"

class RigolChannelCoupling(ExtendedEnum):
    AC = "AC"
    DC = "DC"
    GND = "GND"

class RigolWaveformSources(ExtendedEnum):
    CHAN1 = "CHANnel1"
    CHAN2 = "CHANnel2"
    CHAN3 = "CHANnel3"
    CHAN4 = "CHANnel4"
    MATH = "MATH"

    @classmethod
    def getChannelNumber(cls, eVal):
        match = re.search(r".*([0-9]).*", eVal.value)
        ch = 0
        if match != None and len(match.groups()) > 0:
            ch = int(match.group(1))
        return ch

class RigolWaveformModes(ExtendedEnum):
    NORM = "NORMal"
    MAX = "MAXimum"
    RAW = "RAW"

class RigolWaveformFormats(ExtendedEnum):
    BYTE = "BYTE"
    WORD = "WORD"
    ASC = "ASCii"

class RigolWaveformAxis(ExtendedEnum):
    X = "X"
    Y = "Y"

class RigolWaveformParameter:
    def __init__(self, 
                    format:RigolWaveformFormats = RigolWaveformFormats.BYTE,
                    mode: RigolWaveformModes = RigolWaveformModes.NORM,
                    numPoints:int = 0,
                    avgCount: int = 0,
                    xinc:float = 0.0,
                    xorig:float = 0.0,
                    xref:float = 0.0,
                    yinc:float = 0.0,
                    yorig:float = 0.0,
                    yref:float = 0.0):

        self.format = format
        self.mode = mode
        self.numPoints = numPoints
        self.avgCount = avgCount
        self.xinc = xinc
        self.xorig = xorig
        self.xref = xref
        self.yinc = yinc
        self.yorig = yorig
        self.yref = yref

    def parsePreamble(self, preamble:str):
        data = preamble.split(",")
        
        if len(data) != 10:
            raise ValueError("Preamble string doesn't have 10 elems!")
        
        self.format = RigolWaveformFormats.list()[int(data[0])]
        self.mode = RigolWaveformModes.list()[int(data[1])]
        self.numPoints = int(data[2])
        self.avgCount = int(data[3])
        self.xinc = float(data[4])
        self.xorig = float(data[5])
        self.xref = float(data[6])
        self.yinc = float(data[7])
        self.yorig = float(data[8])
        self.yref = float(data[9])
class RigolVisaDS1000ZE:

    WAVEFORM_DATA_HEADER_LENGTH = 12
    
    X_GRID_ELEMS = 10
    Y_GRID_ELEMS = 8

    TIMEBASE_MIN_S = 2e-9
    TIMEBASE_MAX_S = 50
    TIMEBASE_STEPS = [1, 2.5, 5]

    def __init__(self, addr, maxChannels=2):
        self.addr = addr
        self.channels = maxChannels
        self.instrument = None
        self.id = None
        self.rm = None
        self.timeBaseScales = self.createTimeBaseVector()
    
    def connect(self):
        self.rm = pv.ResourceManager()
        try:
            self.instrument = self.rm.open_resource(self.addr)
        except Exception as e:
            self.instrument = None
        
        if self.instrument != None:
            return True
        else:
            return False 

    def disconnect(self):
        if self.instrument != None:
            self.instrument.close()
        if self.rm != None:
            self.rm.close()
        self.rm = None
        self.instrument = None

        return True

    def writeCmd(self, cmdStr):
        if self.instrument != None:
            try:
                self.instrument.write(cmdStr)
            except Exception as e:
                return False
            return True
        return False
    
    def readCmd(self, cmdStr, decode=True, cutNewLine=True):
        if self.writeCmd(cmdStr) == True:
            data = self.instrument.read_raw()
            if cutNewLine == True:
                data = data[:len(data)-1]
            
            if decode == True:
                return data.decode("utf-8")
            else:
                return data
        else:
            return None

    def createTimeBaseVector(self):
        timebases = list()
        step = 10
        tVal = self.TIMEBASE_MIN_S
        while tVal <= self.TIMEBASE_MAX_S:
            for factor in self.TIMEBASE_STEPS:
                temp = tVal * factor
                if temp <= self.TIMEBASE_MAX_S:
                    timebases.append(round(tVal*factor, 8))
                else:
                    break
            tVal = tVal * step
        return timebases


    def getValidTimeBaseScale(self, tScale):
        timebase = 0
        for i in range(1,len(self.timeBaseScales)):
            if tScale < self.timeBaseScales[i]:
                timebase = self.timeBaseScales[i-1]
                break
        
        if timebase == 0:
            timebase = self.TIMEBASE_MAX_S
        
        return timebase

    def checkValidChannel(self, channel:int):
        if channel <= self.channels:
            return True
        else:
            raise ValueError(f"Channel number {channel} exceeds maximum number of {self.channels} channels!")

    def getInstrumentId(self):
        id = self.readCmd("*IDN?")
        return id

    def startScope(self):
         return self.writeCmd(":RUN")
    
    def stopScope(self):
         return self.writeCmd(":STOP")

    def setAutoScale(self):
         return self.writeCmd(":AUToscale")
    
    def setSingleTrigger(self):
         return self.writeCmd(":SINGle")
    
    def forceTrigger(self):
         return self.writeCmd(":TFORce")

    def setChannelBwLimit(self, channel:int, bwLimit:RigolBwLimit):
        if channel <= self.channels:
             return self.writeCmd(f":CHANnel{channel}:BWLimit {bwLimit.value}")
    
    def getChannelBwLimit(self, channel:int):
        if self.checkValidChannel(channel):
            bwLimit = self.readCmd(f":CHANnel{channel}:BWLimit?")
            bwLimit = RigolBwLimit.getEnumFromStr(bwLimit)
            return bwLimit
        else:
            return None

    def setChannelCoupling(self, channel:int, coupling:RigolChannelCoupling):
        if self.checkValidChannel(channel) == True:
             return self.writeCmd(f":CHANnel{channel}:COUPling {coupling.value}")
      
    def getChannelCoupling(self, channel:int):
        if self.checkValidChannel(channel):
            data = self.readCmd(f":CHANnel{channel}:COUPling?")
            coupling = RigolChannelCoupling.getEnumFromStr(data)
            return coupling
        else:
            return None
    
    def setDisplayChannel(self, channel:int, display:bool):
        if self.checkValidChannel(channel):
            if display == True:
                display = "ON"
            else:
                display = "OFF"

            return self.writeCmd(f":CHANnel{channel}:DISPlay {display}")
    
    def getDisplayChannel(self, channel:int):
        if self.checkValidChannel(channel):
            display = self.readCmd(f":CHANnel{channel}:DISPlay?")
            if int(display) == 0:
                display = False
            else:
                display = True
            return display
        else:
            return None
    
    def setChannelScale(self, channel:int, scale:float):
        if self.checkValidChannel(channel):
            return self.writeCmd(f":CHANnel{channel}:SCALe {scale}")
    
    def getChannelScale(self, channel:int):
        if self.checkValidChannel(channel):
            scale = float(self.readCmd(f":CHANnel{channel}:SCALe?"))
            return scale

    def setTimeBaseScale(self, tScale:float):
        tScale = self.getValidTimeBaseScale(tScale)

        return self.writeCmd(f":TIMebase:SCALe {tScale}")
    
    def getTimeBaseScale(self):
        tScale = float(self.readCmd(":TIMebase:SCALe?"))
        return tScale
    
    def setWaveformSource(self, channel:RigolWaveformSources):
        if self.checkValidChannel(RigolWaveformSources.getChannelNumber(channel)):
            return self.writeCmd(f":WAVeform:SOURce {channel.value}")
    
    def getWaveformSource(self):
        data = self.readCmd(":WAVeform:SOURce?")
        wavSource = RigolWaveformSources[data]
        return wavSource
    
    def setWaveformMode(self, wavMode:RigolWaveformModes):
        return self.writeCmd(f":WAVeform:MODE {wavMode.value}")

    def getWaveformMode(self):
        data = self.readCmd(":WAVeform:MODE?")
        wavMode = RigolWaveformModes[data]
        return wavMode

    def setWaveformFormat(self, wavFmt:RigolWaveformFormats):
        return self.writeCmd(f":WAVeform:FORMat {wavFmt.value}")
    
    def getWaveformFormat(self):
        data = self.readCmd(":WAVeform:FORMat?")
        wavFmt = RigolWaveformFormats[data]
        return wavFmt

    def getWaveformData(self):
        data = self.readCmd(":WAVeform:DATA?", decode=False)
        data = data[self.WAVEFORM_DATA_HEADER_LENGTH:len(data)]

        values = list()
        for i in range(len(data)):
            values.append(float(data[i]))

        return values
    
    def getWaveformFromChannel(self, channel:RigolWaveformSources,
                                mode:RigolWaveformModes,
                                format:RigolWaveformFormats):
        self.setWaveformSource(channel)
        if channel != self.getWaveformSource():
            raise RuntimeError("Waveform source not set correctly!")
        
        self.setWaveformMode(mode)
        if mode != self.getWaveformMode():
            raise RuntimeError("Waveform mode not set correctly!")
        
        self.setWaveformFormat(format)
        if format != self.getWaveformFormat():
            raise RuntimeWarning("Waveform format not set correctly!")
        
        return self.getWaveformData()

    def convertWaveformDataToVolts(self, data):
        params = self.getWaveformParameters()
        volts = list()
        for d in data:
            volt = (d - params.yorig - params.yref) * params.yinc
            volts.append(volt)
        return volts

    def getWaveformIncrement(self, axis:RigolWaveformAxis):
        data = self.readCmd(f":WAVeform:{axis.value}INCrement?")
        inc = float(data)
        return inc
    
    def getWaveformOrigin(self, axis:RigolWaveformAxis):
        data = self.readCmd(f":WAVeform:{axis.value}ORigin?")
        orig = float(data)
        return orig
    
    def getWaveformReference(self, axis:RigolWaveformAxis):
        data = self.readCmd(f":WAVeform:{axis.value}REFerence?")
        ref = float(data)
        return ref
    
    def getWaveformParameters(self):
        data = self.readCmd(":WAVeform:PREamble?")
        params = RigolWaveformParameter()
        params.parsePreamble(data)
        return params

