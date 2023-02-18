import unittest
import numpy as np
from ...tests_common import *
from ..echo_bipolar_bf import EchoBipolarBF

class TestEchoBipolarBF(TestCommon, unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp(EchoBipolarBF)

    def testEncodeDecode42(self):
        echo = EchoBipolarBF(source_int16_len_131072, secret_uint8_42)
        output, additional_output = echo.encode(d0=250, d1=350)

        self.assertEqual(additional_output['l'], secret_uint8_42.size)

        echo = EchoBipolarBF(output)
        output, additional_output = echo.decode(d0=250, d1=350, l=additional_output['l'])
        np.testing.assert_equal(secret_uint8_42, output)
