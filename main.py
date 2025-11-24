import numpy as np
import matplotlib.pyplot as plt

# 1. Cargar la imagen
img = plt.imread("planeta.png")  # Cambia por el nombre de tu archivo

# 2. Convertir a escala de grises si es una imagen RGB
if img.ndim == 3:              # Imagen con 3 canales (R, G, B)
    img_gray = img.mean(axis=2)
else:                          # Ya está en gris
    img_gray = img

# 3. Calcular la Transformada de Fourier 2D
F = np.fft.fft2(img_gray)

# 4. Mover las bajas frecuencias al centro
F_shift = np.fft.fftshift(F)

# 5. Obtener el espectro de magnitud en escala logarítmica
magnitude_spectrum = 20 * np.log10(np.abs(F_shift) + 1e-8)

# 6. Graficar imagen original y su espectro
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Imagen en escala de grises")
plt.imshow(img_gray, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Espectro de Fourier")
plt.imshow(magnitude_spectrum, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

# 7. Reconstruir la imagen con la transformada inversa (opcional)
F_ishift = np.fft.ifftshift(F_shift)
img_recon = np.fft.ifft2(F_ishift)
img_recon = np.real(img_recon)

plt.figure(figsize=(5, 4))
plt.title("Imagen reconstruida")
plt.imshow(img_recon, cmap="gray")
plt.axis("off")
plt.show()