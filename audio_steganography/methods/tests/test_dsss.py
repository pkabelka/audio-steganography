import unittest
import numpy as np
from ...tests_common import *
from ..dsss import DSSS

class TestDSSS(TestCommon, unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp(DSSS)

    def testEncodeDecode42(self):
        echo = DSSS(np.repeat(source_int16_len_131072, 3), secret_uint8_42)
        output, additional_output = echo.encode(password='some password 123')

        self.assertEqual(additional_output['l'], secret_uint8_42.size)

        echo = DSSS(output)
        output, additional_output = echo.decode(
            password='some password 123',
            l=additional_output['l'],
        )
        np.testing.assert_equal(secret_uint8_42, output)
