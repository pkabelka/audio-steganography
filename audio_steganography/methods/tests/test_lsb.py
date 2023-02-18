import unittest
import numpy as np
from ...tests_common import *
from ..lsb import LSB
from ...exceptions import SecretSizeTooLarge

class TestLSB(TestCommon, unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp(LSB)

    def testWrongTypeSource(self):
        lsb = LSB(source_object_empty, secret_uint8_42)
        self.assertRaisesRegex(
            ValueError,
            "Invalid integer data type 'O'.",
            lsb.encode)

    def testEncodeDecode42DepthImplicit(self):
        lsb = LSB(source_int16_len_32, secret_uint8_42)
        output, additional_output = lsb.encode()

        self.assertEqual(output.size, source_int16_len_32.size)
        self.assertEqual(additional_output['l'], secret_uint8_42.size)

        lsb = LSB(output)
        output, additional_output = lsb.decode()
        np.testing.assert_equal(output[:secret_uint8_42.size], secret_uint8_42)

    def testEncodeDecode42Depth1(self):
        lsb = LSB(source_int16_len_32, secret_uint8_42)
        output, additional_output = lsb.encode(depth=1)

        self.assertEqual(output.size, source_int16_len_32.size)
        self.assertEqual(additional_output['l'], secret_uint8_42.size)

        lsb = LSB(output)
        output, additional_output = lsb.decode(depth=1)
        np.testing.assert_equal(output[:secret_uint8_42.size], secret_uint8_42)

    def testEncodeDecode42Depth2(self):
        lsb = LSB(source_int16_len_32, secret_uint8_42)
        output, additional_output = lsb.encode(depth=2)

        self.assertEqual(output.size, source_int16_len_32.size)
        self.assertEqual(additional_output['l'], secret_uint8_42.size)

        lsb = LSB(output)
        output, additional_output = lsb.decode(depth=2)
        np.testing.assert_equal(output[:secret_uint8_42.size], secret_uint8_42)

    def testEncodeDecode42Depth8(self):
        lsb = LSB(source_int16_len_32, secret_uint8_42)
        output, additional_output = lsb.encode(depth=8)

        self.assertEqual(output.size, source_int16_len_32.size)
        self.assertEqual(additional_output['l'], secret_uint8_42.size)

        lsb = LSB(output)
        output, additional_output = lsb.decode(depth=8)
        np.testing.assert_equal(output[:secret_uint8_42.size], secret_uint8_42)

    def testEncodeDecode42Depth9(self):
        lsb = LSB(source_int16_len_32, secret_uint8_42)
        output, additional_output = lsb.encode(depth=9)

        self.assertEqual(output.size, source_int16_len_32.size)
        self.assertEqual(additional_output['l'], secret_uint8_42.size)

        lsb = LSB(output)
        output, additional_output = lsb.decode(depth=9)
        np.testing.assert_equal(output[:secret_uint8_42.size], secret_uint8_42)

    def testEncodeDecode42DepthTooManyBits(self):
        lsb = LSB(source_int16_len_32, secret_uint8_42)
        self.assertRaisesRegex(
            ValueError,
            "bit depth must be between 1 and 16",
            lsb.encode,
            depth=17)

    def testEncodeDecode42DepthMismatch(self):
        lsb = LSB(source_int16_len_32, secret_uint8_42)
        output, additional_output = lsb.encode(depth=1)

        self.assertEqual(output.size, source_int16_len_32.size)
        self.assertEqual(additional_output['l'], secret_uint8_42.size)

        lsb = LSB(output)
        output, additional_output = lsb.decode(depth=2)

        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_equal,
            output[:secret_uint8_42.size],
            secret_uint8_42)
