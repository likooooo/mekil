import numpy as np

a = np.array([             1,  2,  3,  4])
reshaped_a = a.reshape((4,1))
shifted_a = np.fft.fftshift(reshaped_a)
print(shifted_a)