from DSPTools import FirFilter, FirFilterFactory

coeffs = FirFilterFactory.getLowPassCoefficients(2000,3117,49880, N=80)
f = FirFilter(coeffs)

print(coeffs)
print("")
# Turn to true for use with CMSIS DSP lib!
f.printCoeffsCArray(True)

input("Press Enter to exit!")