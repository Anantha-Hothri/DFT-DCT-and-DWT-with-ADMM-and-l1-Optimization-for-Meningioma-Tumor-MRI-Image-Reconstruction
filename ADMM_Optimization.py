import numpy as np
import pywt
from scipy.fftpack import dct, idct
from tqdm import tqdm
from skimage import data
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import time

# ADMM Optimization for compressed sensing
def admm_optimization(A, y, lam, rho, max_iter=100):
    m, n = A.shape
    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)

    def shrinkage(x, kappa):
        return np.maximum(0, x - kappa) - np.maximum(0, -x - kappa)

    for _ in tqdm(range(max_iter), desc="ADMM Optimization Progress"):
        # x-update
        x = np.linalg.solve(A.T @ A + rho * np.eye(n), A.T @ y + rho * (z - u))
        
        # z-update with shrinkage
        z = shrinkage(x + u, lam / rho)
        
        # u-update
        u = u + x - z

    return x

# Compressed Sensing with DWT
def compressed_sensing_dwt(image, wavelet, level, lam, rho):
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    
    m, n = coeff_arr.shape
    A = np.random.randn(int(0.5 * m * n), m * n)
    y = A @ coeff_arr.flatten()

    x_recovered = admm_optimization(A, y, lam, rho)
    coeffs_recovered = pywt.array_to_coeffs(x_recovered.reshape(m, n), coeff_slices, output_format='wavedec2')
    image_recovered = pywt.waverec2(coeffs_recovered, wavelet)

    return image_recovered

# Compressed Sensing with DCT
def compressed_sensing_dct(image, lam, rho):
    coeff_arr = dct(dct(image.T, norm='ortho').T, norm='ortho')
    
    m, n = coeff_arr.shape
    A = np.random.randn(int(0.5 * m * n), m * n)
    y = A @ coeff_arr.flatten()

    x_recovered = admm_optimization(A, y, lam, rho)
    coeff_arr_recovered = x_recovered.reshape(m, n)
    image_recovered = idct(idct(coeff_arr_recovered.T, norm='ortho').T, norm='ortho')

    return image_recovered

# Compressed Sensing with Fourier
def compressed_sensing_fourier(image, lam, rho):
    coeff_arr = np.fft.fft2(image)
    
    m, n = coeff_arr.shape
    A = np.random.randn(int(0.5 * m * n), m * n)
    y = A @ coeff_arr.flatten()

    x_recovered = admm_optimization(A, y, lam, rho)
    coeff_arr_recovered = x_recovered.reshape(m, n)
    image_recovered = np.fft.ifft2(coeff_arr_recovered).real

    return image_recovered

# Load and resize an example image
image_path = r"D:\ASEB\Semester 4\Projects Semester 4\MFC\archive_mt\meningioma_tumor_testing\image(93).jpg"
image = np.mean(plt.imread(image_path), axis=2)
image_resized = resize(image, (128, 128), anti_aliasing=True)

# Parameters
wavelet = 'db1'
level = 3
lam = 0.1
rho = 1.0

# Compressed Sensing with DWT
start_time_dwt = time.time()
image_recovered_dwt = compressed_sensing_dwt(image_resized, wavelet, level, lam, rho)
end_time_dwt = time.time()
dwt_time = end_time_dwt - start_time_dwt
dwt_psnr = psnr(image_resized, image_recovered_dwt)

# Compressed Sensing with DCT
start_time_dct = time.time()
image_recovered_dct = compressed_sensing_dct(image_resized, lam, rho)
end_time_dct = time.time()
dct_time = end_time_dct - start_time_dct
dct_psnr = psnr(image_resized, image_recovered_dct)

# Compressed Sensing with Fourier
start_time_fourier = time.time()
image_recovered_fourier = compressed_sensing_fourier(image_resized, lam, rho)
end_time_fourier = time.time()
fourier_time = end_time_fourier - start_time_fourier
fourier_psnr = psnr(image_resized, image_recovered_fourier)

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
