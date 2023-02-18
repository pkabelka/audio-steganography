import unittest
import numpy as np
from ...tests_common import *
from ..echo_single import EchoSingle

class TestEchoSingle(TestCommon, unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp(EchoSingle)

    def testEncodeDecode42(self):
        echo = EchoSingle(source_int16_len_131072, secret_uint8_42)
        output, additional_output = echo.encode(d0=250, d1=350)

        self.assertEqual(additional_output['l'], secret_uint8_42.size)

        echo = EchoSingle(output)
        output, additional_output = echo.decode(d0=250, d1=350, l=additional_output['l'])
        np.testing.assert_equal(secret_uint8_42, output)
