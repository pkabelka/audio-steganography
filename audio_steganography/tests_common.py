import numpy as np
import unittest
import abc
from typing import Type
from .methods.method_base import MethodBase
from .exceptions import SecretSizeTooLarge


# secret
secret_empty = np.array([])
secret_uint8_empty = np.array([], dtype=np.uint8)
secret_int16_empty = np.array([], dtype=np.int16)
secret_uint8_42 = np.array([0,0,1,1,0,1,0,0,0,0,1,1,0,0,1,0], dtype=np.uint8)
secret_uint16_42 = np.array([0,0,1,1,0,1,0,0,0,0,1,1,0,0,1,0], dtype=np.int16)

# source
## empty sources
source_empty = np.array([])
source_object_empty = np.array([], dtype='object')
source_uint8_empty = np.array([], dtype=np.uint8)
source_int16_empty = np.array([], dtype=np.int16)
source_int32_empty = np.array([], dtype=np.int32)
source_float16_empty = np.array([], dtype=np.float16)
source_float32_empty = np.array([], dtype=np.float32)
source_float64_empty = np.array([], dtype=np.float64)

## various size and type sources
### lenght 1
size = 1
source_uint8_len_1 = np.array(np.random.randint(256, size=size)).astype(np.uint8)
source_int16_len_1 = np.array(np.random.randint(65536, size=size)).astype(np.int16)
source_int32_len_1 = np.array(np.random.randint(2147483648, size=size)).astype(np.int32)
source_float16_len_1 = np.array(np.random.rand(1) * 2 - 1).astype(np.float16)
source_float32_len_1 = np.array(np.random.rand(1) * 2 - 1).astype(np.float32)
source_float64_len_1 = np.array(np.random.rand(1) * 2 - 1).astype(np.float64)

### length 32
size = 32
source_uint8_len_32 = np.array(np.random.randint(256, size=size)).astype(np.uint8)
source_int16_len_32 = np.array(np.random.randint(65536, size=size)).astype(np.int16)
source_int32_len_32 = np.array(np.random.randint(2147483648, size=size)).astype(np.int32)
source_float16_len_32 = np.array(np.random.rand(size) * 2 - 1).astype(np.float16)
source_float32_len_32 = np.array(np.random.rand(size) * 2 - 1).astype(np.float32)
source_float64_len_32 = np.array(np.random.rand(size) * 2 - 1).astype(np.float64)

### length 1024 seed=42
size = 1024
rs = np.random.RandomState(seed=42)
source_uint8_len_1024 = np.array(rs.randint(256, size=size)).astype(np.uint8)
source_int16_len_1024 = np.array(rs.randint(65536, size=size)).astype(np.int16)
source_int1024_len_1024 = np.array(rs.randint(2147483648, size=size)).astype(np.int32)
source_float16_len_1024 = np.array(rs.rand(size) * 2 - 1).astype(np.float16)
source_float32_len_1024 = np.array(rs.rand(size) * 2 - 1).astype(np.float32)
source_float64_len_1024 = np.array(rs.rand(size) * 2 - 1).astype(np.float64)

### length 32768 seed=42
size = 32768
rs = np.random.RandomState(seed=42)
source_uint8_len_32768 = np.array(rs.randint(256, size=size)).astype(np.uint8)
source_int16_len_32768 = np.array(rs.randint(65536, size=size)).astype(np.int16)
source_int32_len_32768 = np.array(rs.randint(2147483648, size=size)).astype(np.int32)
source_float16_len_32768 = np.array(rs.rand(size) * 2 - 1).astype(np.float16)
source_float32_len_32768 = np.array(rs.rand(size) * 2 - 1).astype(np.float32)
source_float64_len_32768 = np.array(rs.rand(size) * 2 - 1).astype(np.float64)

### length 131072 seed=42
size = 131072
rs = np.random.RandomState(seed=42)
source_uint8_len_131072 = np.array(rs.randint(256, size=size)).astype(np.uint8)
source_int16_len_131072 = np.array(rs.randint(65536, size=size)).astype(np.int16)
source_int32_len_131072 = np.array(rs.randint(2147483648, size=size)).astype(np.int32)
source_float16_len_131072 = np.array(rs.rand(size) * 2 - 1).astype(np.float16)
source_float32_len_131072 = np.array(rs.rand(size) * 2 - 1).astype(np.float32)
source_float64_len_131072 = np.array(rs.rand(size) * 2 - 1).astype(np.float64)

### length 524288 seed=42
# size = 524288
# rs = np.random.RandomState(seed=42)
# source_uint8_len_524288 = np.array(rs.randint(256, size=size)).astype(np.uint8)
# source_int16_len_524288 = np.array(rs.randint(65536, size=size)).astype(np.int16)
# source_int32_len_524288 = np.array(rs.randint(2147483648, size=size)).astype(np.int32)
# source_float16_len_524288 = np.array(rs.rand(size) * 2 - 1).astype(np.float16)
# source_float32_len_524288 = np.array(rs.rand(size) * 2 - 1).astype(np.float32)
# source_float64_len_524288 = np.array(rs.rand(size) * 2 - 1).astype(np.float64)



class TestCommon(abc.ABC):
    @abc.abstractmethod
    def setUp(self, method: Type[MethodBase]) -> None:
        self._method = method
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def testWrongTypeSecret(self):
        self.assertRaisesRegex(
            TypeError,
            'secret_data must be of type numpy.uint8',
            self._method,
            source_empty,
            secret_empty)

        self.assertRaisesRegex(
            TypeError,
            'secret_data must be of type numpy.uint8',
            self._method,
            source_empty,
            secret_int16_empty)

        self.assertRaisesRegex(
            TypeError,
            'secret_data must be of type numpy.uint8',
            self._method,
            source_empty,
            secret_uint16_42)

    def testEmptySecretEncode(self):
        method = self._method(source_uint8_len_32, secret_uint8_empty)
        output, additional_output = method.encode()

        self.assertEqual(output.size, source_uint8_len_32.size)
        np.testing.assert_equal(output, source_uint8_len_32)
        self.assertEqual(additional_output['l'], 0)

    def testEmptyEncode(self):
        method = self._method(source_int16_empty, secret_uint8_empty)
        output, additional_output = method.encode()

        self.assertEqual(output.size, source_int16_empty.size)
        np.testing.assert_equal(output, source_int16_empty)
        self.assertEqual(additional_output['l'], 0)

    def testEmptySourceEncode(self):
        method = self._method(source_uint8_empty, secret_uint8_42)
        self.assertRaises(SecretSizeTooLarge, method.encode)

        method = self._method(source_int16_empty, secret_uint8_42)
        self.assertRaises(SecretSizeTooLarge, method.encode)

        method = self._method(source_int32_empty, secret_uint8_42)
        self.assertRaises(SecretSizeTooLarge, method.encode)

        method = self._method(source_float16_empty, secret_uint8_42)
        self.assertRaises(SecretSizeTooLarge, method.encode)

        method = self._method(source_float32_empty, secret_uint8_42)
        self.assertRaises(SecretSizeTooLarge, method.encode)

        method = self._method(source_float64_empty, secret_uint8_42)
        self.assertRaises(SecretSizeTooLarge, method.encode)
