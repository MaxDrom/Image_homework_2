import numpy as np
from scipy.fftpack import fft2, ifft2
import matplotlib.pyplot as plt
n = 100
image = np.random.random((n, n))
fft_image = fft2(image) 
sorted_ind =np.argsort( np.real(fft_image*np.conjugate(fft_image)), axis=None)
x = []
y = []
for k in range(n):
    x.append(k/n*100)
    fft_image[np.unravel_index(sorted_ind[:k], fft_image.shape)] = 0
    new_image = np.real(ifft2(fft_image))
    y.append(np.mean(1-np.abs(image-new_image)/image))

plt.plot(x,y)
plt.xlabel("%")
plt.ylabel("Quality")
plt.show()
