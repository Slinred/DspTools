import unittest

from RigolVisaLib import RigolBwLimit, RigolChannelCoupling, RigolVisaDS1100ZE

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

if __name__ == '__main__':
    unittest.main()