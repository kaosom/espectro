"""
Script principal para visualizar las propiedades de la Transformada de Fourier.
Este archivo contiene toda la visualización (imshow, prints) según los requisitos funcionales.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from fourier_processing import (
    compute_fourier_transform,
    compute_magnitude,
    compute_phase,
    compute_angle,
    shift_frequency_domain,
    apply_linearity_property,
    apply_shift_property_time,
    apply_shift_property_frequency,
    apply_modulation_time,
    apply_modulation_frequency,
    apply_translation_time,
    apply_translation_frequency
)


def select_image_file():
    """
    Abre un diálogo para que el usuario seleccione una imagen.
    
    Returns:
        str: Ruta del archivo seleccionado, o None si se cancela
    """
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal
    
    file_path = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[
            ("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif"),
            ("Todos los archivos", "*.*")
        ]
    )
    
    root.destroy()
    return file_path


def load_image(file_path):
    """
    Carga una imagen desde un archivo y la convierte a escala de grises.
    
    Args:
        file_path: str, ruta al archivo de imagen
        
    Returns:
        numpy array 2D, imagen en escala de grises normalizada [0, 1]
    """
    try:
        img = Image.open(file_path)
        
        # Convertir a escala de grises si es necesario
        if img.mode != 'L':
            img = img.convert('L')
        
        # Convertir a array numpy y normalizar a [0, 1]
        image_array = np.array(img, dtype=np.float64) / 255.0
        
        print(f"Imagen cargada: {file_path}")
        print(f"Dimensiones: {image_array.shape}")
        
        return image_array
    except Exception as e:
        print(f"Error al cargar la imagen: {e}")
        return None


def visualize_original_and_fourier(image):
    """
    Visualiza la imagen original y su transformada de Fourier (magnitud, ángulo y fase).
    """
    print("\n=== Imagen Original y Transformada de Fourier ===")
    
    # Calcular transformada
    ft = compute_fourier_transform(image)
    magnitude = compute_magnitude(ft)
    phase = compute_phase(ft)
    angle = compute_angle(ft)
    
    # Desplazar para visualización
    magnitude_shifted = shift_frequency_domain(magnitude)
    phase_shifted = shift_frequency_domain(phase)
    angle_shifted = shift_frequency_domain(angle)
    
    # Visualizar
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Imagen Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(np.log1p(magnitude_shifted), cmap='gray')
    axes[0, 1].set_title('Transformada de Fourier (Magnitud)')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(phase_shifted, cmap='hsv')
    axes[1, 0].set_title('Fase')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(angle_shifted, cmap='hsv')
    axes[1, 1].set_title('Ángulo')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('1_original_fourier.png', dpi=150, bbox_inches='tight')
    print("Guardado: 1_original_fourier.png")
    plt.close()


def visualize_linearity_property(image1, image2):
    """
    Visualiza la propiedad de linealidad en dominio tiempo y frecuencia.
    """
    print("\n=== Propiedad de Linealidad ===")
    
    alpha, beta = 0.6, 0.4
    combined_time, combined_freq, ft_combined = apply_linearity_property(
        image1, image2, alpha, beta
    )
    
    # Calcular transformadas individuales
    ft1 = compute_fourier_transform(image1)
    ft2 = compute_fourier_transform(image2)
    
    # Desplazar para visualización
    ft_combined_shifted = shift_frequency_domain(compute_magnitude(ft_combined))
    combined_freq_shifted = shift_frequency_domain(compute_magnitude(combined_freq))
    
    # Visualizar dominio tiempo
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(image1, cmap='gray')
    axes[0, 0].set_title(f'Imagen 1 (α={alpha})')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(image2, cmap='gray')
    axes[0, 1].set_title(f'Imagen 2 (β={beta})')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(combined_time, cmap='gray')
    axes[0, 2].set_title('α·f + β·g (Dominio Tiempo)')
    axes[0, 2].axis('off')
    
    # Visualizar dominio frecuencia
    axes[1, 0].imshow(np.log1p(shift_frequency_domain(compute_magnitude(ft1))), cmap='gray')
    axes[1, 0].set_title('F(Imagen 1)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(np.log1p(shift_frequency_domain(compute_magnitude(ft2))), cmap='gray')
    axes[1, 1].set_title('F(Imagen 2)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(np.log1p(combined_freq_shifted), cmap='gray')
    axes[1, 2].set_title('α·F(f) + β·F(g) (Dominio Frecuencia)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('2_linealidad.png', dpi=150, bbox_inches='tight')
    print("Guardado: 2_linealidad.png")
    plt.close()
    
    # Verificar propiedad
    diff = np.abs(ft_combined - combined_freq)
    print(f"Diferencia máxima entre F(α·f + β·g) y α·F(f) + β·F(g): {np.max(diff):.2e}")


def visualize_shift_property(image):
    """
    Visualiza la propiedad de recorrido/shift en dominio tiempo y frecuencia.
    """
    print("\n=== Propiedad de Recorrido/Shift ===")
    
    shift_x, shift_y = 50, 30
    
    # Shift en dominio tiempo
    shifted_time, ft_shifted_time = apply_shift_property_time(image, shift_x, shift_y)
    
    # Shift en dominio frecuencia
    ft_original = compute_fourier_transform(image)
    shifted_freq, reconstructed = apply_shift_property_frequency(ft_original, shift_x, shift_y)
    
    # Visualizar
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Dominio tiempo
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Imagen Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(shifted_time, cmap='gray')
    axes[0, 1].set_title(f'Desplazada en Tiempo ({shift_x}, {shift_y})')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(np.log1p(shift_frequency_domain(compute_magnitude(ft_shifted_time))), cmap='gray')
    axes[0, 2].set_title('F(Imagen Desplazada)')
    axes[0, 2].axis('off')
    
    # Dominio frecuencia
    axes[1, 0].imshow(np.log1p(shift_frequency_domain(compute_magnitude(ft_original))), cmap='gray')
    axes[1, 0].set_title('F(Imagen Original)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(np.log1p(shift_frequency_domain(compute_magnitude(shifted_freq))), cmap='gray')
    axes[1, 1].set_title(f'F Desplazada en Frecuencia ({shift_x}, {shift_y})')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(reconstructed, cmap='gray')
    axes[1, 2].set_title('Imagen Reconstruida')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('3_recorrido.png', dpi=150, bbox_inches='tight')
    print("Guardado: 3_recorrido.png")
    plt.close()


def visualize_modulation_property(image):
    """
    Visualiza la propiedad de modulación en dominio tiempo y frecuencia.
    """
    print("\n=== Propiedad de Modulación ===")
    
    freq_u, freq_v = 0.1, 0.15
    
    # Modulación en dominio tiempo
    modulated_time, ft_modulated_time = apply_modulation_time(image, freq_u, freq_v)
    
    # Modulación en dominio frecuencia
    ft_original = compute_fourier_transform(image)
    modulated_freq, reconstructed = apply_modulation_frequency(ft_original, freq_u, freq_v)
    
    # Visualizar
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Dominio tiempo
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Imagen Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(modulated_time, cmap='gray')
    axes[0, 1].set_title(f'Modulada en Tiempo (u={freq_u}, v={freq_v})')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(np.log1p(shift_frequency_domain(compute_magnitude(ft_modulated_time))), cmap='gray')
    axes[0, 2].set_title('F(Imagen Modulada)')
    axes[0, 2].axis('off')
    
    # Dominio frecuencia
    axes[1, 0].imshow(np.log1p(shift_frequency_domain(compute_magnitude(ft_original))), cmap='gray')
    axes[1, 0].set_title('F(Imagen Original)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(np.log1p(shift_frequency_domain(compute_magnitude(modulated_freq))), cmap='gray')
    axes[1, 1].set_title(f'F Modulada en Frecuencia (u={freq_u}, v={freq_v})')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(reconstructed, cmap='gray')
    axes[1, 2].set_title('Imagen Reconstruida')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('4_modulacion.png', dpi=150, bbox_inches='tight')
    print("Guardado: 4_modulacion.png")
    plt.close()


def visualize_translation_property(image):
    """
    Visualiza la propiedad de desplazamiento/traslación en dominio tiempo y frecuencia.
    """
    print("\n=== Propiedad de Desplazamiento/Traslación ===")
    
    tx, ty = 40, 60
    
    # Traslación en dominio tiempo
    translated_time, ft_translated_time = apply_translation_time(image, tx, ty)
    
    # Traslación en dominio frecuencia
    ft_original = compute_fourier_transform(image)
    translated_freq, reconstructed = apply_translation_frequency(ft_original, tx, ty)
    
    # Visualizar
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Dominio tiempo
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Imagen Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(translated_time, cmap='gray')
    axes[0, 1].set_title(f'Trasladada en Tiempo ({tx}, {ty})')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(np.log1p(shift_frequency_domain(compute_magnitude(ft_translated_time))), cmap='gray')
    axes[0, 2].set_title('F(Imagen Trasladada)')
    axes[0, 2].axis('off')
    
    # Dominio frecuencia
    axes[1, 0].imshow(np.log1p(shift_frequency_domain(compute_magnitude(ft_original))), cmap='gray')
    axes[1, 0].set_title('F(Imagen Original)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(np.log1p(shift_frequency_domain(compute_magnitude(translated_freq))), cmap='gray')
    axes[1, 1].set_title(f'F Trasladada en Frecuencia ({tx}, {ty})')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(reconstructed, cmap='gray')
    axes[1, 2].set_title('Imagen Reconstruida')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('5_desplazamiento.png', dpi=150, bbox_inches='tight')
    print("Guardado: 5_desplazamiento.png")
    plt.close()


def main():
    """
    Función principal que ejecuta todas las visualizaciones.
    """
    print("=" * 60)
    print("Procesamiento de Imágenes - Transformada de Fourier")
    print("Programación Funcional")
    print("=" * 60)
    
    # Solicitar al usuario que seleccione una imagen
    print("\nPor favor, selecciona una imagen para procesar...")
    file_path = select_image_file()
    
    if file_path is None or file_path == "":
        print("No se seleccionó ninguna imagen. Saliendo...")
        return
    
    # Cargar la imagen seleccionada
    image1 = load_image(file_path)
    
    if image1 is None:
        print("Error al cargar la imagen. Saliendo...")
        return
    
    # Crear una segunda imagen para la propiedad de linealidad
    # Usaremos una versión desplazada/escalada de la imagen original
    rows, cols = image1.shape
    image2 = np.zeros_like(image1)
    # Crear un rectángulo en el centro como segunda imagen
    h, w = rows // 4, cols // 4
    start_row, start_col = rows // 2 - h // 2, cols // 2 - w // 2
    image2[start_row:start_row+h, start_col:start_col+w] = 1.0
    
    # Visualizar todas las propiedades
    visualize_original_and_fourier(image1)
    visualize_linearity_property(image1, image2)
    visualize_shift_property(image1)
    visualize_modulation_property(image1)
    visualize_translation_property(image1)
    
    print("\n" + "=" * 60)
    print("Todas las visualizaciones completadas.")
    print("=" * 60)


if __name__ == "__main__":
    main()

