import unittest
import numpy as np
from ..test import *
from .lsb import LSB
from ..exceptions import SecretSizeTooLarge

class TestLSB(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def testWrongTypeSecret(self):
        self.assertRaisesRegex(
            TypeError,
            'secret_data must be of type numpy.uint8',
            LSB,
            source_empty,
            secret_empty)

        self.assertRaisesRegex(
            TypeError,
            'secret_data must be of type numpy.uint8',
            LSB,
            source_empty,
            secret_int16_empty)

        self.assertRaisesRegex(
            TypeError,
            'secret_data must be of type numpy.uint8',
            LSB,
            source_empty,
            secret_uint16_42)

    def testEmptySecretEncode(self):
        zeroed_lsb_reference = np.bitwise_and(source_uint8_len_32, np.bitwise_not(1))

        lsb = LSB(source_uint8_len_32, secret_uint8_empty)
        output, additional_output = lsb.encode()

        self.assertEqual(output.size, source_uint8_len_32.size)
        np.testing.assert_equal(output, zeroed_lsb_reference)
        self.assertEqual(additional_output['l'], secret_uint8_empty.size)

    def testWrongTypeSource(self):
        lsb = LSB(source_object_empty, secret_uint8_empty)
        self.assertRaisesRegex(
            ValueError,
            "Invalid integer data type 'O'.",
            lsb.encode)

    def testEmptyEncode(self):
        lsb = LSB(source_int16_empty, secret_uint8_empty)
        output, additional_output = lsb.encode()

        self.assertEqual(output.size, source_int16_empty.size)
        np.testing.assert_equal(output, source_int16_empty)
        self.assertEqual(additional_output['l'], secret_uint8_empty.size)

    def testEmptySourceEncode(self):
        lsb = LSB(source_uint8_empty, secret_uint8_42)
        self.assertRaises(SecretSizeTooLarge, lsb.encode)

        lsb = LSB(source_int16_empty, secret_uint8_42)
        self.assertRaises(SecretSizeTooLarge, lsb.encode)

        lsb = LSB(source_int32_empty, secret_uint8_42)
        self.assertRaises(SecretSizeTooLarge, lsb.encode)

        lsb = LSB(source_float16_empty, secret_uint8_42)
        self.assertRaises(SecretSizeTooLarge, lsb.encode)

        lsb = LSB(source_float32_empty, secret_uint8_42)
        self.assertRaises(SecretSizeTooLarge, lsb.encode)

        lsb = LSB(source_float64_empty, secret_uint8_42)
        self.assertRaises(SecretSizeTooLarge, lsb.encode)

    def testEncodeDecode42(self):
        lsb = LSB(source_int16_len_32, secret_uint8_42)
        output, additional_output = lsb.encode()

        self.assertEqual(output.size, source_int16_len_32.size)
        self.assertEqual(additional_output['l'], secret_uint8_42.size)

        lsb = LSB(output)
        output, additional_output = lsb.decode()
        np.testing.assert_equal(output[:secret_uint8_42.size], secret_uint8_42)
