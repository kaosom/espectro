"""
Módulo de procesamiento de imágenes usando programación funcional.
Todas las funciones son puras: no modifican sus argumentos y siempre devuelven nuevas matrices.
"""

import numpy as np


def compute_fourier_transform(image):
    """
    Calcula la Transformada de Fourier 2D de una imagen.
    
    Esta función es pura y devuelve la transformada completa (compleja).
    Se usa para obtener la representación en frecuencia de la imagen,
    permitiendo analizar las componentes espectrales.
    
    Args:
        image: numpy array 2D, imagen en escala de grises
        
    Returns:
        numpy array 2D complejo, transformada de Fourier
    """
    return np.fft.fft2(image)


def compute_inverse_fourier_transform(fourier_transform):
    """
    Calcula la Transformada Inversa de Fourier 2D.
    
    Esta función es pura y convierte la representación en frecuencia
    de vuelta al dominio espacial. Se usa para reconstruir la imagen
    después de operaciones en el dominio de frecuencia.
    
    Args:
        fourier_transform: numpy array 2D complejo, transformada de Fourier
        
    Returns:
        numpy array 2D real, imagen reconstruida
    """
    return np.fft.ifft2(fourier_transform).real


def compute_magnitude(fourier_transform):
    """
    Calcula la magnitud (espectro de potencia) de la transformada de Fourier.
    
    Esta función es pura y extrae la magnitud de cada componente frecuencial.
    Se usa para visualizar la distribución de energía en el dominio de frecuencia,
    mostrando qué frecuencias están presentes en la imagen.
    
    Args:
        fourier_transform: numpy array 2D complejo, transformada de Fourier
        
    Returns:
        numpy array 2D real, magnitud de la transformada
    """
    return np.abs(fourier_transform)


def compute_phase(fourier_transform):
    """
    Calcula la fase de la transformada de Fourier.
    
    Esta función es pura y extrae la información de fase de cada componente.
    La fase contiene información sobre la posición y estructura espacial
    de las componentes frecuenciales en la imagen original.
    
    Args:
        fourier_transform: numpy array 2D complejo, transformada de Fourier
        
    Returns:
        numpy array 2D real, fase en radianes
    """
    return np.angle(fourier_transform)


def compute_angle(fourier_transform):
    """
    Calcula el ángulo de la transformada de Fourier (equivalente a fase).
    
    Esta función es pura y proporciona el ángulo de cada componente compleja.
    Se usa como alternativa a compute_phase para visualización,
    representando la dirección de cada vector complejo en el plano.
    
    Args:
        fourier_transform: numpy array 2D complejo, transformada de Fourier
        
    Returns:
        numpy array 2D real, ángulo en radianes
    """
    return np.angle(fourier_transform)


def shift_frequency_domain(fourier_transform):
    """
    Desplaza el origen de la transformada de Fourier al centro de la imagen.
    
    Esta función es pura y reorganiza los coeficientes de Fourier para que
    las frecuencias bajas queden en el centro. Se usa para visualización,
    ya que es más intuitivo ver el espectro centrado.
    
    Args:
        fourier_transform: numpy array 2D complejo, transformada de Fourier
        
    Returns:
        numpy array 2D complejo, transformada desplazada
    """
    return np.fft.fftshift(fourier_transform)


def unshift_frequency_domain(shifted_fourier):
    """
    Revierte el desplazamiento del origen de la transformada de Fourier.
    
    Esta función es pura y restaura el orden original de los coeficientes
    después de operaciones en el dominio de frecuencia centrado.
    Se usa antes de aplicar la transformada inversa.
    
    Args:
        shifted_fourier: numpy array 2D complejo, transformada desplazada
        
    Returns:
        numpy array 2D complejo, transformada sin desplazamiento
    """
    return np.fft.ifftshift(shifted_fourier)


def apply_linearity_property(image1, image2, alpha, beta):
    """
    Demuestra la propiedad de linealidad de la Transformada de Fourier.
    
    Esta función es pura y verifica que F(α·f + β·g) = α·F(f) + β·F(g).
    Se usa para comprobar que la transformada es un operador lineal,
    lo cual es fundamental para el procesamiento de señales.
    
    Args:
        image1: numpy array 2D, primera imagen
        image2: numpy array 2D, segunda imagen
        alpha: float, coeficiente para la primera imagen
        beta: float, coeficiente para la segunda imagen
        
    Returns:
        tuple: (combinación en tiempo, combinación en frecuencia, transformada de combinación)
    """
    # Combinación lineal en dominio tiempo
    combined_time = alpha * image1 + beta * image2
    
    # Transformada de la combinación
    ft_combined = compute_fourier_transform(combined_time)
    
    # Combinación lineal en dominio frecuencia
    ft_image1 = compute_fourier_transform(image1)
    ft_image2 = compute_fourier_transform(image2)
    combined_freq = alpha * ft_image1 + beta * ft_image2
    
    return combined_time, combined_freq, ft_combined


def apply_shift_property_time(image, shift_x, shift_y):
    """
    Aplica desplazamiento en el dominio tiempo y calcula su efecto en frecuencia.
    
    Esta función es pura y demuestra que un desplazamiento en el dominio espacial
    resulta en una modulación de fase en el dominio de frecuencia.
    Se usa para verificar la propiedad: F(f(x-x0, y-y0)) = F(f)·exp(-j2π(u·x0 + v·y0)).
    
    Args:
        image: numpy array 2D, imagen original
        shift_x: int, desplazamiento en dirección x
        shift_y: int, desplazamiento en dirección y
        
    Returns:
        tuple: (imagen desplazada, transformada de la imagen desplazada)
    """
    rows, cols = image.shape
    
    # Crear matriz de desplazamiento (sin modificar la original)
    shifted = np.zeros_like(image)
    
    # Calcular índices desplazados
    for i in range(rows):
        for j in range(cols):
            new_i = (i - shift_y) % rows
            new_j = (j - shift_x) % cols
            shifted[new_i, new_j] = image[i, j]
    
    # Transformada de la imagen desplazada
    ft_shifted = compute_fourier_transform(shifted)
    
    return shifted, ft_shifted


def apply_shift_property_frequency(fourier_transform, shift_u, shift_v):
    """
    Aplica desplazamiento en el dominio frecuencia y calcula su efecto en tiempo.
    
    Esta función es pura y demuestra que un desplazamiento en el dominio de frecuencia
    resulta en una modulación en el dominio espacial.
    Se usa para verificar la propiedad dual: desplazamiento en frecuencia
    causa modulación en tiempo.
    
    Args:
        fourier_transform: numpy array 2D complejo, transformada de Fourier
        shift_u: int, desplazamiento en frecuencia u
        shift_v: int, desplazamiento en frecuencia v
        
    Returns:
        tuple: (transformada desplazada, imagen reconstruida)
    """
    rows, cols = fourier_transform.shape
    
    # Desplazar en frecuencia (sin modificar la original)
    shifted_ft = np.zeros_like(fourier_transform, dtype=complex)
    
    # Calcular índices desplazados
    for i in range(rows):
        for j in range(cols):
            new_i = (i - shift_v) % rows
            new_j = (j - shift_u) % cols
            shifted_ft[new_i, new_j] = fourier_transform[i, j]
    
    # Reconstruir imagen en dominio tiempo
    reconstructed = compute_inverse_fourier_transform(shifted_ft)
    
    return shifted_ft, reconstructed


def apply_modulation_time(image, frequency_u, frequency_v):
    """
    Aplica modulación en el dominio tiempo y calcula su efecto en frecuencia.
    
    Esta función es pura y demuestra que una modulación (multiplicación por coseno)
    en el dominio espacial resulta en un desplazamiento en el dominio de frecuencia.
    Para imágenes reales, usamos modulación coseno: f(x,y)·cos(2π(u0·x + v0·y)).
    Se usa para verificar la propiedad de modulación de la Transformada de Fourier.
    
    Args:
        image: numpy array 2D, imagen original
        frequency_u: float, frecuencia de modulación en dirección u
        frequency_v: float, frecuencia de modulación en dirección v
        
    Returns:
        tuple: (imagen modulada, transformada de la imagen modulada)
    """
    rows, cols = image.shape
    
    # Crear matriz de modulación (sin modificar la original)
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    
    # Modulación coseno (para imágenes reales)
    modulation = np.cos(2 * np.pi * (frequency_u * X / cols + frequency_v * Y / rows))
    
    # Aplicar modulación (multiplicación elemento a elemento, crea nueva matriz)
    modulated = image * modulation
    
    # Transformada de la imagen modulada
    ft_modulated = compute_fourier_transform(modulated)
    
    return modulated, ft_modulated


def apply_modulation_frequency(fourier_transform, frequency_u, frequency_v):
    """
    Aplica modulación en el dominio frecuencia y calcula su efecto en tiempo.
    
    Esta función es pura y demuestra que una modulación en el dominio de frecuencia
    resulta en un desplazamiento en el dominio espacial.
    Se usa para verificar la propiedad dual de la modulación.
    
    Args:
        fourier_transform: numpy array 2D complejo, transformada de Fourier
        frequency_u: float, frecuencia de modulación en dirección u
        frequency_v: float, frecuencia de modulación en dirección v
        
    Returns:
        tuple: (transformada modulada, imagen reconstruida)
    """
    rows, cols = fourier_transform.shape
    
    # Crear matriz de modulación en frecuencia
    u = np.arange(cols)
    v = np.arange(rows)
    U, V = np.meshgrid(u, v)
    
    # Modulación compleja en frecuencia
    modulation = np.exp(1j * 2 * np.pi * (frequency_u * U / cols + frequency_v * V / rows))
    
    # Aplicar modulación (multiplicación elemento a elemento)
    modulated_ft = fourier_transform * modulation
    
    # Reconstruir imagen en dominio tiempo
    reconstructed = compute_inverse_fourier_transform(modulated_ft)
    
    return modulated_ft, reconstructed


def apply_translation_time(image, tx, ty):
    """
    Aplica traslación (desplazamiento circular) en el dominio tiempo.
    
    Esta función es pura y desplaza la imagen circularmente sin modificar la original.
    Se usa para demostrar el efecto de la traslación en el dominio espacial
    y su correspondiente cambio de fase en frecuencia.
    
    Args:
        image: numpy array 2D, imagen original
        tx: int, traslación en dirección x
        ty: int, traslación en dirección y
        
    Returns:
        tuple: (imagen trasladada, transformada de la imagen trasladada)
    """
    # Usar roll para desplazamiento circular (crea nueva matriz)
    translated = np.roll(image, (ty, tx), axis=(0, 1))
    
    # Transformada de la imagen trasladada
    ft_translated = compute_fourier_transform(translated)
    
    return translated, ft_translated


def apply_translation_frequency(fourier_transform, tu, tv):
    """
    Aplica traslación (desplazamiento circular) en el dominio frecuencia.
    
    Esta función es pura y desplaza la transformada circularmente sin modificar la original.
    Se usa para demostrar el efecto de la traslación en el dominio de frecuencia
    y su correspondiente modulación en tiempo.
    
    Args:
        fourier_transform: numpy array 2D complejo, transformada de Fourier
        tu: int, traslación en dirección u
        tv: int, traslación en dirección v
        
    Returns:
        tuple: (transformada trasladada, imagen reconstruida)
    """
    # Usar roll para desplazamiento circular (crea nueva matriz)
    translated_ft = np.roll(fourier_transform, (tv, tu), axis=(0, 1))
    
    # Reconstruir imagen en dominio tiempo
    reconstructed = compute_inverse_fourier_transform(translated_ft)
    
    return translated_ft, reconstructed

