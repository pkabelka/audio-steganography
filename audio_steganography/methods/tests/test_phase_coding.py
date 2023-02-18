import unittest
import numpy as np
from ...tests_common import *
from ..phase_coding import PhaseCoding

class TestPhaseCoding(TestCommon, unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp(PhaseCoding)

    def testEncodeDecode42(self):
        echo = PhaseCoding(source_int16_len_1024, secret_uint8_42)
        output, additional_output = echo.encode()

        self.assertEqual(additional_output['l'], secret_uint8_42.size)

        echo = PhaseCoding(output)
        output, additional_output = echo.decode(l=additional_output['l'])
        np.testing.assert_equal(secret_uint8_42, output)
