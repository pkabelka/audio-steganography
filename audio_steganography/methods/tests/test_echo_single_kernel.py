import unittest
import numpy as np
from ...test import *
from ..echo_single_kernel import Echo_single_kernel
from ...exceptions import SecretSizeTooLarge
from ...stat_utils import ber_percent

class TestLSB(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def testWrongTypeSecret(self):
        self.assertRaisesRegex(
            TypeError,
            'secret_data must be of type numpy.uint8',
            Echo_single_kernel,
            source_empty,
            secret_empty)

        self.assertRaisesRegex(
            TypeError,
            'secret_data must be of type numpy.uint8',
            Echo_single_kernel,
            source_empty,
            secret_int16_empty)

        self.assertRaisesRegex(
            TypeError,
            'secret_data must be of type numpy.uint8',
            Echo_single_kernel,
            source_empty,
            secret_uint16_42)

    def testEmptySecretEncode(self):
        echo = Echo_single_kernel(source_uint8_len_32, secret_uint8_empty)
        output, additional_output = echo.encode()

        self.assertEqual(output.size, source_uint8_len_32.size)
        np.testing.assert_equal(output, source_uint8_len_32)
        self.assertEqual(additional_output['l'], secret_uint8_empty.size)

    def testEmptyEncode(self):
        echo = Echo_single_kernel(source_int16_empty, secret_uint8_empty)
        output, additional_output = echo.encode()

        self.assertEqual(output.size, source_int16_empty.size)
        np.testing.assert_equal(output, source_int16_empty)
        self.assertEqual(additional_output['l'], secret_uint8_empty.size)

    def testEmptySourceEncode(self):
        echo = Echo_single_kernel(source_uint8_empty, secret_uint8_42)
        self.assertRaises(SecretSizeTooLarge, echo.encode)

        echo = Echo_single_kernel(source_int16_empty, secret_uint8_42)
        self.assertRaises(SecretSizeTooLarge, echo.encode)

        echo = Echo_single_kernel(source_int32_empty, secret_uint8_42)
        self.assertRaises(SecretSizeTooLarge, echo.encode)

        echo = Echo_single_kernel(source_float16_empty, secret_uint8_42)
        self.assertRaises(SecretSizeTooLarge, echo.encode)

        echo = Echo_single_kernel(source_float32_empty, secret_uint8_42)
        self.assertRaises(SecretSizeTooLarge, echo.encode)

        echo = Echo_single_kernel(source_float64_empty, secret_uint8_42)
        self.assertRaises(SecretSizeTooLarge, echo.encode)

    def testEncodeDecode42(self):
        echo = Echo_single_kernel(source_int16_len_131072, secret_uint8_42)
        output, additional_output = echo.encode(d0=2306, d1=3000)

        self.assertEqual(additional_output['l'], secret_uint8_42.size)

        echo = Echo_single_kernel(output)
        output, additional_output = echo.decode(d0=250, d1=350, l=additional_output['l'])
        self.assertEqual(ber_percent(secret_uint8_42, output), 31.25)
