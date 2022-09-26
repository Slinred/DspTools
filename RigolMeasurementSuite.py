

from math import log10, sqrt


class MeasurementResult:
    def __init__(self, name, values:list, unit:str):
        self.name = name
        self.values = values
        self.unit = unit

class RigolMeasurementSuite():
    def __init__(self):
        self.freqResp = None

    def calcRmsValue(self, data):
        eff = max(data) / sqrt(2)
        return eff

    def calcDecibels(self, val):
        dbVal = 20 * log10(val)
        return dbVal

    def measureFrequencyResponse(self, frequencyVector, getDataCbk, filterFunction = None):
        print("--------------------------------------------------------------------------------")
        print("- Measuring frequency response...")
        
        resultRms = list()
        resultDB = list()

        for freq in frequencyVector:
            print(f"  -> Measuring f={freq} Hz ...")
            print("    |-> Getting values...")
            data = getDataCbk()
            print(f"        |-> Got {len(data)} values!")
            if len(data) > 0:
                if filterFunction != None:
                    print("    |-> Applying filter...")
                    data = filterFunction(data)

                print("    |-> Calculating RMS...")
                rmsVal = self.calcRmsValue(data)
                print(f"        |-> RMS = {rmsVal}")
                resultRms.append(rmsVal)

                print("    |-> Calculating logarithmic value...")
                dbVal = self.calcDecibels(rmsVal)
                print(f"        |-> dB = {dbVal}")
                resultDB.append(dbVal)
            else:
                raise ValueError("No data received!")
            print("")
        
        rmsValResult = MeasurementResult("RMS", resultRms, "V")
        dbValResult = MeasurementResult("Amplitude", resultDB, "dB")

        print("- DONE")
        print("--------------------------------------------------------------------------------")

        return (rmsValResult, dbValResult)
        
