import numpy as np
import pywt
from scipy.fftpack import dct, idct
from tqdm import tqdm
from skimage import io
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import time

# l1 Optimization for compressed sensing with progress bar
def l1_optimization(A, y, max_iter=100):
    m, n = A.shape
    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)

    def shrinkage(x, kappa):
        return np.maximum(0, x - kappa) - np.maximum(0, -x - kappa)

    for _ in tqdm(range(max_iter), desc="l1 Optimization Progress"):
        # x-update
        x = np.linalg.solve(A.T @ A + np.eye(n), A.T @ y + (z - u))
        
        # z-update with shrinkage
        z = shrinkage(x + u, 1)
        
        # u-update
        u = u + x - z

    return x

# Compressed Sensing with DWT
def compressed_sensing_dwt(image, wavelet, level):
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    
    m, n = coeff_arr.shape
    A = np.random.randn(int(0.5 * m * n), m * n)
    y = A @ coeff_arr.flatten()

    x_recovered = l1_optimization(A, y)
    coeffs_recovered = pywt.array_to_coeffs(x_recovered.reshape(m, n), coeff_slices, output_format='wavedec2')
    image_recovered = pywt.waverec2(coeffs_recovered, wavelet)

    return image_recovered

# Compressed Sensing with DCT
def compressed_sensing_dct(image):
    coeff_arr = dct(dct(image.T, norm='ortho').T, norm='ortho')
    
    m, n = coeff_arr.shape
    A = np.random.randn(int(0.5 * m * n), m * n)
    y = A @ coeff_arr.flatten()

    x_recovered = l1_optimization(A, y)
    coeff_arr_recovered = x_recovered.reshape(m, n)
    image_recovered = idct(idct(coeff_arr_recovered.T, norm='ortho').T, norm='ortho')

    return image_recovered

# Compressed Sensing with Fourier
def compressed_sensing_fourier(image):
    coeff_arr = np.fft.fft2(image)
    
    m, n = coeff_arr.shape
    A = np.random.randn(int(0.5 * m * n), m * n)
    y = A @ coeff_arr.flatten()

    x_recovered = l1_optimization(A, y)
    coeff_arr_recovered = x_recovered.reshape(m, n)
    image_recovered = np.fft.ifft2(coeff_arr_recovered).real

    return image_recovered

# Load and resize the uploaded image
image = io.imread(r"C:\Users\Advik Narendran\Downloads\MFC project\archive_mt\archive_mt\meningioma_tumor_training\m3 (94).jpg", as_gray=True)
image_resized = resize(image, (128, 128), anti_aliasing=True)

# Ensure the pixel values are within the range [0, 1]
image_resized = (image_resized - image_resized.min()) / (image_resized.max() - image_resized.min())

# Parameters
wavelet = 'db1'
level = 3

# Compressed Sensing with DWT
start_time_dwt = time.time()
image_recovered_dwt = compressed_sensing_dwt(image_resized, wavelet, level)
end_time_dwt = time.time()
dwt_time = end_time_dwt - start_time_dwt
dwt_psnr = psnr(image_resized, image_recovered_dwt, data_range=image_resized.max() - image_resized.min())

# Compressed Sensing with DCT
start_time_dct = time.time()
image_recovered_dct = compressed_sensing_dct(image_resized)
end_time_dct = time.time()
dct_time = end_time_dct - start_time_dct
dct_psnr = psnr(image_resized, image_recovered_dct, data_range=image_resized.max() - image_resized.min())

# Compressed Sensing with Fourier
start_time_fourier = time.time()
image_recovered_fourier = compressed_sensing_fourier(image_resized)
end_time_fourier = time.time()
fourier_time = end_time_fourier - start_time_fourier
fourier_psnr = psnr(image_resized, image_recovered_fourier, data_range=image_resized.max() - image_resized.min())

# Print the results
print(f"DWT - Time: {dwt_time:.2f} seconds, PSNR: {dwt_psnr:.2f} dB")
print(f"DCT - Time: {dct_time:.2f} seconds, PSNR: {dct_psnr:.2f} dB")
print(f"Fourier - Time: {fourier_time:.2f} seconds, PSNR: {fourier_psnr:.2f} dB")

# Plot results
plt.figure(figsize=(20, 5))
plt.subplot(1, 5, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 5, 2)
plt.title("Resized Image")
plt.imshow(image_resized, cmap='gray')
plt.axis('off')

plt.subplot(1, 5, 3)
plt.title(f"Recovered Image (DWT)\nTime: {dwt_time:.2f}s\nPSNR: {dwt_psnr:.2f} dB")
plt.imshow(image_recovered_dwt, cmap='gray')
plt.axis('off')

plt.subplot(1, 5, 4)
plt.title(f"Recovered Image (DCT)\nTime: {dct_time:.2f}s\nPSNR: {dct_psnr:.2f} dB")
plt.imshow(image_recovered_dct, cmap='gray')
plt.axis('off')

plt.subplot(1, 5, 5)
plt.title(f"Recovered Image (Fourier)\nTime: {fourier_time:.2f}s\nPSNR: {fourier_psnr:.2f} dB")
plt.imshow(image_recovered_fourier, cmap='gray')
plt.axis('off')

plt.show()

#3:01:09+ 3 min 17 sec
#= 3:03:26