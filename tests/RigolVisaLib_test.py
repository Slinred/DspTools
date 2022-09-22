import unittest

from RigolVisaLib import RigolBwLimit, RigolChannelCoupling, RigolVisaDS1100ZE, RigolWaveformSources

SCOPE_ADDR = "TCPIP::172.16.1.121::INSTR"

class TestRigolVisaLib(unittest.TestCase):

    def test_RigolDS1100ZE_connect(self):
        scope = RigolVisaDS1100ZE(SCOPE_ADDR)
        self.assertTrue(scope.connect(), "Checking connection")
        self.assertTrue(scope.disconnect())

    def test_RigolDS1100ZE_getId(self):
        scope = RigolVisaDS1100ZE(SCOPE_ADDR)
        self.assertTrue(scope.connect(), "Checking connection")
        self.assertEqual(scope.getInstrumentId(), "RIGOL TECHNOLOGIES,DS1102Z-E,DS1ZE224309861,00.06.02")
        self.assertTrue(scope.disconnect())

    def test_RigolDS1100ZE_channelValid(self):
        scope = RigolVisaDS1100ZE(SCOPE_ADDR)

        with self.assertRaises(ValueError):
            scope.checkValidChannel(3)
        self.assertTrue(scope.disconnect())
    
    def test_RigolDS1100ZE_BwLimit(self):
        scope = RigolVisaDS1100ZE(SCOPE_ADDR)
        self.assertTrue(scope.connect())
        
        for bwLimit in RigolBwLimit.list():
            bwLimit = RigolBwLimit.ON_20MHz
            scope.setChannelBwLimit(1, bwLimit)
            self.assertEqual(bwLimit, scope.getChannelBwLimit(1))

        self.assertTrue(scope.disconnect())

    def test_RigolDS1100ZE_Coupling(self):
        scope = RigolVisaDS1100ZE(SCOPE_ADDR)
        self.assertTrue(scope.connect())

        for c in RigolChannelCoupling.list():
            scope.setChannelCoupling(1, c)
            self.assertEqual(c, scope.getChannelCoupling(1))

        self.assertTrue(scope.disconnect())
    
    def test_RigolDS1100ZE_DisplayChannel(self):
        scope = RigolVisaDS1100ZE(SCOPE_ADDR)
        self.assertTrue(scope.connect())

        for d in [False, True]:
            scope.setDisplayChannel(1, d)
            self.assertEqual(d, scope.getDisplayChannel(1))

        self.assertTrue(scope.disconnect())    

    def test_RigolDS1100ZE_ChannelScale(self):
        scope = RigolVisaDS1100ZE(SCOPE_ADDR)
        self.assertTrue(scope.connect())
        
        scales = [0.1, 1, 2, 5, 10]

        for s in scales:
            scope.setChannelScale(1, s)
            self.assertEqual(s, scope.getChannelScale(1))

        self.assertTrue(scope.disconnect())  

    def test_RigolDS1100ZE_TimebaseScale(self):
        scope = RigolVisaDS1100ZE(SCOPE_ADDR)
        self.assertTrue(scope.connect())
        
        tBases = [0.0001, 0.001, 0.002, 0.05, 0.1]

        for t in tBases:
            scope.setTimeBaseScale(t)
            self.assertEqual(t, scope.getTimeBaseScale())

        self.assertTrue(scope.disconnect())  

    def test_RigolDS1100ZE_WaveformSource(self):
        scope = RigolVisaDS1100ZE(SCOPE_ADDR)
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

if __name__ == '__main__':
    unittest.main()