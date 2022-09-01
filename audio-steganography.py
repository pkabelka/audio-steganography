import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import scipy.signal
import sys
# import os

# np.set_printoptions(threshold=sys.maxsize)

message = 'msg'
encode = []
for c in message:
    for b in '{0:08b}'.format(ord(c), 'b'):
        encode.append(int(b))
encode = np.array(encode)
print(message)
print(encode)

def seg_split(sig, nseg):
    return np.array_split(sig, nseg)[:-1] + [sig[-int(round(len(sig)/nseg)):]]

sr, s = scipy.io.wavfile.read('speech.wav')

print('Signal len:', len(s))

c = len(encode)
mixer = seg_split(np.ones(len(s)), c+1)
print('Mixer len:', len(mixer))
print('Seg sample len:', len(mixer[0]))

for i in range(len(encode)):
    mixer[i] = mixer[i] * encode[i]

mixer = np.hstack(mixer)
print(mixer)
print('Mixer len:', len(mixer))

delay_pairs = []
end = False
for d0 in range(250, 350):
    h0 = np.append(np.zeros(d0), [1])
    for d1 in range(d0, d0+100):

        # d0 = 105

        # d1 = 110
        h1 = np.append(np.zeros(d1), [1])

        k0 = scipy.signal.fftconvolve(h0, s)
        k1 = scipy.signal.fftconvolve(h1, s)

        sp = np.pad(np.array(s), (0, len(k1)-len(s)))
        x = np.zeros(len(sp))
        x = sp[:len(mixer)]+k1[:len(mixer)] * mixer + sp[:len(mixer)]+k0[:len(mixer)] * (1-mixer)

        x_f = x - np.mean(x)
        x_f = x_f / np.abs(x_f).max()
        # scipy.io.wavfile.write('echo.wav', sr, x_f)

        decoded = []
        split = seg_split(x_f, c+1)[:-1]
        for seg in split:
            cn = np.fft.ifft(np.log(np.abs(np.fft.fft(seg))))
            if cn[d0+1] > cn[d1+1]:
                decoded.append(0)
            else:
                decoded.append(1)

        decoded = np.array(decoded)
        print(decoded)
        decoded_message = ''
        for i in range(0, len(decoded), 8):
            decoded_message += chr(int(''.join([str(x) for x in decoded[i:i+8]]), 2))
        print(decoded_message)

        if np.all(decoded == encode):
            delay_pairs.append((d0, d1))
            end = True
            break
    if end:
        break

print(delay_pairs)
