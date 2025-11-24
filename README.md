# Procesamiento de Imágenes - Transformada de Fourier

Este proyecto implementa el procesamiento de imágenes usando programación funcional para demostrar las propiedades de la Transformada de Fourier.

## Requisitos

- Python 3.7+
- numpy
- matplotlib

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
python main.py
```

El script generará 5 archivos de visualización:
1. `1_original_fourier.png` - Imagen original y transformada de Fourier (magnitud, fase, ángulo)
2. `2_linealidad.png` - Propiedad de linealidad
3. `3_recorrido.png` - Propiedad de recorrido/shift
4. `4_modulacion.png` - Propiedad de modulación
5. `5_desplazamiento.png` - Propiedad de desplazamiento/traslación

## Estructura del Proyecto

- `fourier_processing.py`: Módulo con funciones puras de procesamiento (sin efectos secundarios)
- `main.py`: Script principal con toda la visualización
- `requirements.txt`: Dependencias del proyecto

## Principios de Programación Funcional

Todas las funciones en `fourier_processing.py` son puras:
- No modifican sus argumentos
- Siempre devuelven nuevas matrices
- No tienen efectos secundarios (no imprimen, no hacen gráficas)
- No usan variables globales
- No usan operaciones in-place

