from enum import Enum
import re
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

class RigolVisaDS1100ZE:
    def __init__(self, addr, maxChannels=2):
        self.addr = addr
        self.channels = maxChannels
        self.instrument = None
        self.id = None
        self.rm = None
    
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