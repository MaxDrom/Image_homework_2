from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift, ifft2
from scipy.ndimage import convolve

def do_plot(name, data):
    fig, ax = plt.subplots()
    ax.imshow(1000*np.abs(data))
    fig.set_figwidth(data.shape[0]/100)   
    fig.set_figheight(data.shape[1]/100)    
    #plt.show()
    plt.savefig(name)

def mean_nearest(x, y, data):
    result = 0
    for i in range(x-1, x+2):
        for j in range(y-1, y+2):
            if (i-x)**2+(j-y)**2 != 0:
                result+= data[i, j]
    return result

with fits.open("noised.fits") as hdu_list:
    data = hdu_list[0].data

fourier_img = fft2(data) 
fourier_shifted = fftshift(fourier_img)
do_plot("fft.png",np.abs(fourier_shifted))

kernel = 1/8*np.ones((3,3))
kernel[1,1] = 0
four_mean = convolve(fourier_img, kernel)
difference = fourier_img - four_mean
four_mean_sd = convolve(difference*np.conjugate(difference),kernel)
four_mean_sd = np.sqrt(np.real(four_mean_sd))

(n, m) = data.shape
for i in range(n):
    for j in range(m):
        if abs(fourier_img[i, j] - four_mean[i,j])> 5*four_mean_sd[i,j]:
            fourier_img[i,j] = four_mean[i,j]

fourier_shifted = fftshift(fourier_img)
do_plot("fft_res.png",np.abs(fourier_shifted))

new_data = ifft2(fourier_img)

hdu = fits.PrimaryHDU(data=np.real(new_data))
hdu_list = fits.HDUList([hdu])
hdu_list.writeto('result.fits', overwrite=True)
