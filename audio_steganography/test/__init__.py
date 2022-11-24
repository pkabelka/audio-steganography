import numpy as np

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
source_float16_len_32 = np.array(np.random.rand(32) * 2 - 1).astype(np.float16)
source_float32_len_32 = np.array(np.random.rand(32) * 2 - 1).astype(np.float32)
source_float64_len_32 = np.array(np.random.rand(32) * 2 - 1).astype(np.float64)
