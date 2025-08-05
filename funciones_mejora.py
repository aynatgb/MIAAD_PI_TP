# funciones_mejora.py
import cv2
import numpy as np

def aplicar_ecualizacion_histograma(imagen_original_bgr):
    """
    Aplica la ecualización de histograma tradicional a una imagen.
    Convierte la imagen a escala de grises si es a color.
    Retorna la imagen ecualizada en escala de grises.

    Args:
        imagen_original_bgr (numpy.ndarray): Imagen original en formato BGR (o escala de grises).

    Returns:
        numpy.ndarray: Imagen ecualizada en escala de grises.
    """
    # Convertir a escala de grises si la imagen es a color
    if len(imagen_original_bgr.shape) == 3:
        imagen_gris = cv2.cvtColor(imagen_original_bgr, cv2.COLOR_BGR2GRAY)
    else:
        imagen_gris = imagen_original_bgr # Ya está en escala de grises

    # Aplicar ecualización de histograma
    imagen_ecualizada = cv2.equalizeHist(imagen_gris)
    print("Ecualización de Histograma (HE) aplicada.")
    return imagen_ecualizada

def aplicar_clahe(imagen_original_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) a una imagen.
    Convierte la imagen a escala de grises si es a color.
    Retorna la imagen ecualizada con CLAHE en escala de grises.

    Args:
        imagen_original_bgr (numpy.ndarray): Imagen original en formato BGR (o escala de grises).
        clip_limit (float): Umbral para limitar la amplificación del contraste.
        tile_grid_size (tuple): Tamaño de la cuadrícula sobre la cual se aplica la ecualización.

    Returns:
        numpy.ndarray: Imagen ecualizada con CLAHE en escala de grises.
    """
    # Convertir a escala de grises si la imagen es a color
    if len(imagen_original_bgr.shape) == 3:
        imagen_gris = cv2.cvtColor(imagen_original_bgr, cv2.COLOR_BGR2GRAY)
    else:
        imagen_gris = imagen_original_bgr

    # Crear un objeto CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Aplicar CLAHE
    imagen_clahe = clahe.apply(imagen_gris)
    print(f"CLAHE aplicada (clipLimit={clip_limit}, tileGridSize={tile_grid_size}).")
    return imagen_clahe

def tu_algoritmo_adicional(imagen_original_bgr):
    """
    Implementa aquí tu algoritmo adicional de mejora de imagen.
    Este es un placeholder. Debes reemplazarlo con la lógica de tu algoritmo.
    Por ahora, simplemente devuelve una copia de la imagen original en gris.

    Args:
        imagen_original_bgr (numpy.ndarray): Imagen original en formato BGR (o escala de grises).

    Returns:
        numpy.ndarray: Imagen mejorada por tu algoritmo (puede ser gris o BGR).
    """
    # Ejemplo de placeholder: simplemente devuelve la imagen en escala de grises
    if len(imagen_original_bgr.shape) == 3:
        imagen_gris = cv2.cvtColor(imagen_original_bgr, cv2.COLOR_BGR2GRAY)
    else:
        imagen_gris = imagen_original_bgr
    
    print("Algoritmo adicional (placeholder) aplicado. ¡Implementa tu lógica aquí!")
    return imagen_gris.copy() # Devuelve una copia para evitar modificar el original

if __name__ == "__main__":
    # Ejemplo de uso si este script se ejecuta directamente
    # Crear una imagen de ejemplo (BGR)
    ejemplo_img_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    ejemplo_img_bgr[20:80, 20:80] = [50, 100, 150] # Un cuadrado de color
    ejemplo_img_bgr[40:60, 40:60] = [10, 20, 30] # Un cuadrado más oscuro

    # Aplicar HE
    img_he = aplicar_ecualizacion_histograma(ejemplo_img_bgr)
    cv2.imshow("Original", ejemplo_img_bgr)
    cv2.imshow("HE", img_he)

    # Aplicar CLAHE
    img_clahe = aplicar_clahe(ejemplo_img_bgr)
    cv2.imshow("CLAHE", img_clahe)

    # Aplicar el algoritmo adicional (placeholder)
    img_adicional = tu_algoritmo_adicional(ejemplo_img_bgr)
    cv2.imshow("Algoritmo Adicional", img_adicional)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
