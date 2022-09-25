import unittest

from RigolVisaLib import RigolBwLimit, RigolChannelCoupling, RigolVisaDS1000ZE, RigolWaveformAxis, RigolWaveformFormats, RigolWaveformModes, RigolWaveformParameter, RigolWaveformSources

SCOPE_ADDR = "TCPIP::172.16.1.121::INSTR"

class TestRigolVisaDS1000ZE(unittest.TestCase):

    def test_RigolDS1000ZE_connect(self):
        scope = RigolVisaDS1000ZE(SCOPE_ADDR)
        self.assertTrue(scope.connect(), "Checking connection")
        self.assertTrue(scope.disconnect())

    def test_RigolDS1000ZE_getId(self):
        scope = RigolVisaDS1000ZE(SCOPE_ADDR)
        self.assertTrue(scope.connect(), "Checking connection")
        self.assertEqual(scope.getInstrumentId(), 
                        "RIGOL TECHNOLOGIES,DS1102Z-E,DS1ZE224309861,00.06.02")
        self.assertTrue(scope.disconnect())

    def test_RigolDS1000ZE_channelValid(self):
        scope = RigolVisaDS1000ZE(SCOPE_ADDR)

        with self.assertRaises(ValueError):
            scope.checkValidChannel(3)
        self.assertTrue(scope.disconnect())
    
    def test_RigolDS1000ZE_BwLimit(self):
        scope = RigolVisaDS1000ZE(SCOPE_ADDR)
        self.assertTrue(scope.connect())
        
        for bwLimit in RigolBwLimit.list():
            bwLimit = RigolBwLimit.ON_20MHz
            scope.setChannelBwLimit(1, bwLimit)
            self.assertEqual(bwLimit, scope.getChannelBwLimit(1))

        self.assertTrue(scope.disconnect())

    def test_RigolDS1000ZE_Coupling(self):
        scope = RigolVisaDS1000ZE(SCOPE_ADDR)
        self.assertTrue(scope.connect())

        for c in RigolChannelCoupling.list():
            scope.setChannelCoupling(1, c)
            self.assertEqual(c, scope.getChannelCoupling(1))

        self.assertTrue(scope.disconnect())
    
    def test_RigolDS1000ZE_DisplayChannel(self):
        scope = RigolVisaDS1000ZE(SCOPE_ADDR)
        self.assertTrue(scope.connect())

        for d in [False, True]:
            scope.setDisplayChannel(1, d)
            self.assertEqual(d, scope.getDisplayChannel(1))

        self.assertTrue(scope.disconnect())    

    def test_RigolDS1000ZE_ChannelScale(self):
        scope = RigolVisaDS1000ZE(SCOPE_ADDR)
        self.assertTrue(scope.connect())
        
        scales = [0.1, 1, 2, 5, 10]

        for s in scales:
            scope.setChannelScale(1, s)
            self.assertEqual(s, scope.getChannelScale(1))

        self.assertTrue(scope.disconnect())  

    def test_RigolDS1000ZE_TimebaseScale(self):
        scope = RigolVisaDS1000ZE(SCOPE_ADDR)
        self.assertTrue(scope.connect())
        
        tBases = [0.0001, 0.001, 0.00375, 0.05, 0.1]

        for t in tBases:
            scope.setTimeBaseScale(t)
            try:
                scope.timeBaseScales.index(t)
                self.assertEqual(t, scope.getTimeBaseScale())
            except ValueError:
                self.assertEqual(scope.getValidTimeBaseScale(t), scope.getTimeBaseScale())

        self.assertTrue(scope.disconnect())  

    def test_RigolDS1000ZE_WaveformSource(self):
        scope = RigolVisaDS1000ZE(SCOPE_ADDR)
        self.assertTrue(scope.connect())

        for ws in RigolWaveformSources.list():
            ch = RigolWaveformSources.getChannelNumber(ws)

            if ch <= scope.channels:
                scope.setWaveformSource(ws)
                self.assertEqual(ws, scope.getWaveformSource())
            else:
                with self.assertRaises(ValueError):
                    scope.setWaveformSource(ws)
                    scope.getWaveformSource()

        self.assertTrue(scope.disconnect()) 
    
    def test_RigolDS1000ZE_WaveformMode(self):
        scope = RigolVisaDS1000ZE(SCOPE_ADDR)
        self.assertTrue(scope.connect())

        scope.setWaveformSource(RigolWaveformSources.CHAN1)

        for wm in RigolWaveformModes.list():
            scope.setWaveformMode(wm)
            self.assertEqual(wm, scope.getWaveformMode())

        self.assertTrue(scope.disconnect()) 
    
    def test_RigolDS1000ZE_WaveformFormat(self):
        scope = RigolVisaDS1000ZE(SCOPE_ADDR)
        self.assertTrue(scope.connect())

        scope.setWaveformSource(RigolWaveformSources.CHAN1)

        for wf in RigolWaveformFormats.list():
            scope.setWaveformFormat(wf)
            self.assertEqual(wf, scope.getWaveformFormat())

        self.assertTrue(scope.disconnect()) 
    
    def test_RigolDS1000ZE_WaveformData(self):
        scope = RigolVisaDS1000ZE(SCOPE_ADDR)
        self.assertTrue(scope.connect())
        
        scope.startScope()
        scope.setChannelCoupling(1, RigolChannelCoupling.DC)
        self.assertEqual(RigolChannelCoupling.DC, scope.getChannelCoupling(1))

        scope.setWaveformSource(RigolWaveformSources.CHAN1)
        self.assertEqual(RigolWaveformSources.CHAN1, scope.getWaveformSource())

        scope.setWaveformMode(RigolWaveformModes.NORM)
        self.assertEqual(RigolWaveformModes.NORM, scope.getWaveformMode())

        scope.setWaveformFormat(RigolWaveformFormats.BYTE)
        self.assertEqual(RigolWaveformFormats.BYTE, scope.getWaveformFormat())
        
        scope.stopScope()

        values = scope.getWaveformData()

        self.assertLess(0, len(values))

        scope.startScope()

        self.assertTrue(scope.disconnect()) 

    def test_RigolDS1000ZE_WaveformIncrement(self):
        scope = RigolVisaDS1000ZE(SCOPE_ADDR)
        self.assertTrue(scope.connect())

        scope.setWaveformSource(RigolWaveformSources.CHAN1)

        for ax in RigolWaveformAxis.list():
            self.assertNotEqual(0, scope.getWaveformIncrement(ax))

        self.assertTrue(scope.disconnect()) 

    def test_RigolDS1000ZE_WaveformReference(self):
        scope = RigolVisaDS1000ZE(SCOPE_ADDR)
        self.assertTrue(scope.connect())

        scope.setWaveformSource(RigolWaveformSources.CHAN1)

        for ax in RigolWaveformAxis.list():
            self.assertNotEqual(None, scope.getWaveformReference(ax))

        self.assertTrue(scope.disconnect()) 
    
    def test_RigolDS1000ZE_WaveformOrigin(self):
        scope = RigolVisaDS1000ZE(SCOPE_ADDR)
        self.assertTrue(scope.connect())

        scope.setWaveformSource(RigolWaveformSources.CHAN1)

        for ax in RigolWaveformAxis.list():
            self.assertNotEqual(None, scope.getWaveformOrigin(ax))

        self.assertTrue(scope.disconnect()) 

    def test_RigolDS1000ZE_WaveformParameters(self):
        preamble = "1,2,6000000,1,1.000000e-09,-3.000000e-03,0,4.132813e-01,0,122"
        
        scope = RigolVisaDS1000ZE(SCOPE_ADDR)
        self.assertTrue(scope.connect())

        params = RigolWaveformParameter()
        params.parsePreamble(preamble)

        self.assertEqual(RigolWaveformFormats.WORD, params.format)
        self.assertEqual(RigolWaveformModes.RAW, params.mode)
        self.assertEqual(6000000, params.numPoints)
        self.assertEqual(1, params.avgCount)
        self.assertEqual(1.000000e-09, params.xinc)
        self.assertEqual(-3.000000e-03, params.xorig)
        self.assertEqual(0, params.xref)
        self.assertEqual(4.132813e-01, params.yinc)
        self.assertEqual(0, params.yorig)
        self.assertEqual(122, params.yref)

        self.assertTrue(scope.disconnect()) 

    def test_RigolDS1000ZE_GetWaveformParams(self):
        scope = RigolVisaDS1000ZE(SCOPE_ADDR)
        self.assertTrue(scope.connect())

        scope.setWaveformSource(RigolWaveformSources.CHAN1)

        self.assertNotEqual(None, scope.getWaveformParameters())

        self.assertTrue(scope.disconnect()) 


if __name__ == '__main__':
    unittest.main()