from .method_base import MethodBase
import numpy as np
import scipy.signal

def seg_split(sig, nseg):
    return np.array_split(sig, nseg)[:-1] + [sig[-int(round(len(sig)/nseg)):]]

class Echo_single_kernel(MethodBase):
    def encode(self) -> np.ndarray:
        self.secret_len = len(self.data_to_encode)
        mixer = seg_split(np.ones(len(self.cover)), self.secret_len+1)
        # print('Mixer len:', len(mixer))
        # print('Seg sample len:', len(mixer[0]))

        for i in range(len(self.data_to_encode)):
            mixer[i] = mixer[i] * self.data_to_encode[i]

        mixer = np.hstack(mixer)
        # print(mixer)
        # print('Mixer len:', len(mixer))

        delay_pairs = []
        end = False
        x = np.array([])
        for d0 in range(250, 350):
            h0 = np.append(np.zeros(d0), [1])
            for d1 in range(d0, d0+100):

                h1 = np.append(np.zeros(d1), [1])

                k0 = scipy.signal.fftconvolve(h0, self.cover)
                k1 = scipy.signal.fftconvolve(h1, self.cover)

                sp = np.pad(np.array(self.cover), (0, len(k1)-len(self.cover)))
                x = np.zeros(len(sp))
                x = sp[:len(mixer)]+k1[:len(mixer)] * mixer + sp[:len(mixer)]+k0[:len(mixer)] * (1-mixer)

                x_f = x - np.mean(x)
                x_f = x_f / np.abs(x_f).max()
                # scipy.io.wavfile.write('echo.wav', sr, x_f)

                # Decode to verify delay pair
                self.data_to_decode = x_f
                self.d0 = d0
                self.d1 = d1

                if np.all(self.decode() == self.data_to_encode):
                    delay_pairs.append((d0, d1))
                    end = True
                    break

            if end:
                break

        print('d0 and d1:', delay_pairs)
        return x


    def decode(self) -> np.ndarray:
        decoded = []
        split = seg_split(self.data_to_decode, self.secret_len+1)[:-1]
        for seg in split:
            cn = np.fft.ifft(np.log(np.abs(np.fft.fft(seg))))
            if cn[self.d0+1] > cn[self.d1+1]:
                decoded.append(0)
            else:
                decoded.append(1)

        decoded = np.array(decoded)
        # print(decoded)
        decoded_message = ''
        for i in range(0, len(decoded), 8):
            decoded_message += chr(int(''.join([str(x) for x in decoded[i:i+8]]), 2))
        # print(decoded_message)

        return(decoded)
