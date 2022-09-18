from genericpath import exists
import sys
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from DSPTools import DspTools, Quantization, TimeSignal, FirFilterFactory

def createCModule(sig:TimeSignal, outDir="."):
    modulePath = f"{outDir}{os.sep}{sig.name}{os.sep}"

    if not exists(modulePath):
        print(f"'{modulePath}' not existsing! Creating...")
        os.mkdir(modulePath)


    headline =     "/*\n" \
                    "* This file was automatically generated\n" \
                    f"* {sig.f} Hz sine sampled @ fs={sig.fs} Hz" + "\n" \
                    "*/\n" \
                    "\n"

    funGetSignalStr =  f"const int16_t * {sig.name}_GetSample(uint16_t idx)"
    numSamplesStr =    f"{sig.name.upper()}_NUM_SAMPLES"

    print(f"Creating header file '{modulePath}{sig.name}.h'...")

    with open(f"{modulePath}{sig.name}.h", "w") as f:
        f.write(    headline + \
                    f"#ifndef {sig.name.upper()}__H" + "\n"
                    f"#define {sig.name.upper()}__H" + "\n"
                    "\n"
                    "#include <stdint.h>\n"
                    "\n"
                    "\n"
                    f"#define {numSamplesStr}   ({len(sig.y)}U)" + "\n"
                    "\n"
                    f"#define {sig.name.upper()}_FS_HZ    ({np.uint16(sig.fs)}U)" + "\n"
                    f"#define {sig.name.upper()}_F_HZ     ({np.uint16(sig.f)}U)" + "\n"
                    "\n"
                    "\n"
                    f"{funGetSignalStr};" + "\n"
                    "\n"
                    "#endif\n"
                    "\n"
                )

    print(f"Header file '{modulePath}{sig.name}.h' created!")
    print(f"Creating source file '{modulePath}{sig.name}.c'...")

    with open(f"{modulePath}{sig.name}.c", "w") as f:
        f.write(headline)
        f.write(
                    f'#include "{sig.name}.h"' + "\n"
                    "\n"
                    "\n"
                )

        f.write(f"const int16_t {sig.name}_Samples[{numSamplesStr}] = " + "\n")
        f.write("  {\n")
        for i in range(0,len(sig.y), 8):
            line = "    "
            for j in range(8):
                if (i+j) < len(sig.y):
                    line += f"{sig.y[i+j]}, "
                line.removesuffix(", ")
            f.write(line + "\n")
        f.write("  };\n\n\n")

        f.write(
                    f"{funGetSignalStr}" + "\n{\n  return " + f"&{sig.name}_Samples[idx]" + ";\n}\n"
                )

    print(f"Source file '{modulePath}{sig.name}.c' created!")

def main(args):

    #args = ["--fSig", "18", "--numSamples", "4096", "--periods", "3"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--fSig", type=int, required=True)
    parser.add_argument("--fs", type=int, required=False, default=0)
    parser.add_argument("--numSamples", type=int, required=False, default=0)
    parser.add_argument("--periods", type=int, required=False, default=1)
    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--outDir", type=str, default=".")
    parsedArgs = parser.parse_args(args)

    if parsedArgs.fs == 0 and parsedArgs.numSamples == 0:
        print("Sample freq or num samples not provided!\nPlease use either --fs to set the sample rate or --numSamples to set the number of samples for this signal!\n")
        parser.print_help()
        exit(0)

    fs = parsedArgs.fs
    n = parsedArgs.numSamples
    periods = parsedArgs.periods

    if periods != 1 and fs > 0 and n > 0:
        raise ValueError("Please specifiy only fs or n when setting periods!")
    elif fs == 0 and n > 0:
        fs = (n * parsedArgs.fSig) / periods
    elif n == 0 and fs > 0:
        n = ((1/parsedArgs.fSig)/(1/fs)) * periods
    else:
        periods = (n*parsedArgs.fSig) / fs

    n = np.uint16(n)
    fs = np.uint16(fs)

    if parsedArgs.fSig >= (fs/2):
        raise ValueError(f"Nyquist rule violated! {parsedArgs.fSig} Hz >= {fs} Hz !")

    print("Creating signal with\n\t"
            f"f  = {parsedArgs.fSig} Hz""\n\t"
            f"fs = {fs} Hz""\n\t"
            f"p  = {periods} periods""\n\t"
            f"n  = {n} samples""\n")

    sig = DspTools.createSineSignal(f"SineTable_{parsedArgs.fSig}Hz", parsedArgs.fSig, fs, periods)
    sig.y = Quantization.quantize(sig.y, Quantization.Levels.Q15, 0)

    if parsedArgs.quiet == False:
        plt.plot(sig.x,sig.y, "-*")
        plt.title(f"Created signal with f={sig.f} Hz @ fs={sig.fs} Hz and {len(sig.y)} samples")
        plt.grid(True)
        plt.show()

    createCModule(sig, parsedArgs.outDir)

    print(f"Created module '{parsedArgs.outDir}{os.sep}{sig.name}'!")
    #input("Press enter to exit!")

if __name__ == "__main__":
    main(sys.argv[1:])