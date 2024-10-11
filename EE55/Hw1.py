from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

output_folder = r'C:\Users\farha\Desktop\Caltech\2024-2025\EE55'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
image_path = r'C:\Users\farha\Desktop\Caltech\2024-2025\EE55\boat.tiff'
img = Image.open(image_path).convert('L')  
img_array = np.array(img)  

plt.imshow(img_array, cmap='gray')
plt.title('Original Image')
plt.colorbar()
plt.show()

original_image_path = os.path.join(output_folder, 'original_image.png')
plt.imsave(original_image_path, img_array, cmap='gray')

quantized_img = np.where(img_array < 127, 0, 255)
plt.imshow(quantized_img, cmap='gray')
plt.title('Quantized Image')
plt.colorbar()
plt.show()

quantized_image_path = os.path.join(output_folder, 'quantized_image.png')
plt.imsave(quantized_image_path, quantized_img, cmap='gray')

def add_noise_and_quantize(image, variance):
    mean = 0  
    noisy_image = image + np.random.normal(mean, np.sqrt(variance), image.shape)
    noisy_image_clipped = np.clip(noisy_image, 0, 255)  
    quantized_noisy_image = np.where(noisy_image_clipped < 127, 0, 255) 
    return quantized_noisy_image

def apply_green_tint(image, variance):
    green_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    green_image[:, :, 0] = image  
    green_image[:, :, 1] = np.clip(image + variance * 5, 0, 255) 
    green_image[:, :, 2] = image  
    return green_image

variances = [0.5, 5, 50]
for variance in variances:
    noisy_quantized_img = add_noise_and_quantize(img_array, variance)
    
    green_tinted_img = apply_green_tint(noisy_quantized_img, variance)
    
    plt.imshow(green_tinted_img)
    plt.title(f'Quantized Image with Noise and Green Tint (Variance={variance})')
    plt.colorbar()
    plt.show()
    
    noisy_image_path = os.path.join(output_folder, f'noisy_quantized_image_variance_{variance}.png')
    plt.imsave(noisy_image_path, green_tinted_img)
