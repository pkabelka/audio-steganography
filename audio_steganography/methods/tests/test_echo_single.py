import unittest
import numpy as np
from ...tests_common import *
from ..echo_single import EchoSingle
from ...exceptions import SecretSizeTooLarge
from ...stat_utils import ber_percent

class TestEchoSingle(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def testWrongTypeSecret(self):
        self.assertRaisesRegex(
            TypeError,
            'secret_data must be of type numpy.uint8',
            EchoSingle,
            source_empty,
            secret_empty)

        self.assertRaisesRegex(
            TypeError,
            'secret_data must be of type numpy.uint8',
            EchoSingle,
            source_empty,
            secret_int16_empty)

        self.assertRaisesRegex(
            TypeError,
            'secret_data must be of type numpy.uint8',
            EchoSingle,
            source_empty,
            secret_uint16_42)

    def testEmptySecretEncode(self):
        echo = EchoSingle(source_uint8_len_32, secret_uint8_empty)
        output, additional_output = echo.encode()

        self.assertEqual(output.size, source_uint8_len_32.size)
        np.testing.assert_equal(output, source_uint8_len_32)
        self.assertEqual(additional_output['l'], -1)

    def testEmptyEncode(self):
        echo = EchoSingle(source_int16_empty, secret_uint8_empty)
        output, additional_output = echo.encode()

        self.assertEqual(output.size, source_int16_empty.size)
        np.testing.assert_equal(output, source_int16_empty)
        self.assertEqual(additional_output['l'], -1)

    def testEmptySourceEncode(self):
        echo = EchoSingle(source_uint8_empty, secret_uint8_42)
        self.assertRaises(SecretSizeTooLarge, echo.encode)

        echo = EchoSingle(source_int16_empty, secret_uint8_42)
        self.assertRaises(SecretSizeTooLarge, echo.encode)

        echo = EchoSingle(source_int32_empty, secret_uint8_42)
        self.assertRaises(SecretSizeTooLarge, echo.encode)

        echo = EchoSingle(source_float16_empty, secret_uint8_42)
        self.assertRaises(SecretSizeTooLarge, echo.encode)

        echo = EchoSingle(source_float32_empty, secret_uint8_42)
        self.assertRaises(SecretSizeTooLarge, echo.encode)

        echo = EchoSingle(source_float64_empty, secret_uint8_42)
        self.assertRaises(SecretSizeTooLarge, echo.encode)

    def testEncodeDecode42(self):
        echo = EchoSingle(source_int16_len_131072, secret_uint8_42)
        output, additional_output = echo.encode(d0=250, d1=350)

        self.assertEqual(additional_output['l'], secret_uint8_42.size)

        echo = EchoSingle(output)
        output, additional_output = echo.decode(d0=250, d1=350, l=additional_output['l'])
        np.testing.assert_equal(secret_uint8_42, output)
