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


def aplicar_dsihe(imagen_original_bgr):
    """
    Aplica el algoritmo de ecualización de histograma de sub-imágenes dualistas
    de área igual (DSIHE).
    El método divide la imagen en dos sub-imágenes de igual área (cantidad de píxeles)
    basado en el valor de la mediana, y luego ecualiza cada sub-imagen
    independientemente.
    Retorna la imagen mejorada en escala de grises.

    Args:
        imagen_original_bgr (numpy.ndarray): Imagen original en formato BGR (o escala de grises).

    Returns:
        numpy.ndarray: Imagen mejorada con DSIHE en escala de grises.
    """
    # Convertir a escala de grises si la imagen es a color
    if len(imagen_original_bgr.shape) == 3:
        imagen_gris = cv2.cvtColor(imagen_original_bgr, cv2.COLOR_BGR2GRAY)
    else:
        imagen_gris = imagen_original_bgr

    # 1. Obtener la mediana del histograma para dividir la imagen en dos sub-imágenes de igual área
    hist = cv2.calcHist([imagen_gris], [0], None, [256], [0, 256]).flatten()
    cdf = hist.cumsum()
    total_pixeles = imagen_gris.size
    mediana_pixel_count = total_pixeles // 2
    
    # Encontrar el nivel de gris que corresponde a la mediana (donde cdf alcanza la mitad de los píxeles)
    umbral_mediana = 0
    for i, count in enumerate(cdf):
        if count >= mediana_pixel_count:
            umbral_mediana = i
            break

    # 2. Segmentar la imagen en dos sub-imágenes: una oscura y otra brillante
    imagen_oscura = imagen_gris[imagen_gris < umbral_mediana]
    imagen_brillante = imagen_gris[imagen_gris >= umbral_mediana]

    # Crear una imagen vacía del mismo tamaño para el resultado
    imagen_dsihe = np.zeros_like(imagen_gris)

    # 3. Aplicar ecualización de histograma a cada sub-imagen por separado
    if imagen_oscura.size > 0:
        hist_oscura = cv2.calcHist([imagen_oscura], [0], None, [umbral_mediana], [0, umbral_mediana]).flatten()
        cdf_oscura = hist_oscura.cumsum()
        
        # Función de transformación para la sub-imagen oscura
        transform_oscura = cdf_oscura / cdf_oscura.max() * (umbral_mediana - 1)
        transform_oscura = transform_oscura.astype('uint8')
        
        # Aplicar la transformación a los píxeles de la imagen oscura
        p_oscura = imagen_gris < umbral_mediana
        imagen_dsihe[p_oscura] = transform_oscura[imagen_gris[p_oscura]]

    if imagen_brillante.size > 0:
        hist_brillante = cv2.calcHist([imagen_brillante], [0], None, [256 - umbral_mediana], [umbral_mediana, 256]).flatten()
        cdf_brillante = hist_brillante.cumsum()
        
        # Función de transformación para la sub-imagen brillante
        transform_brillante = cdf_brillante / cdf_brillante.max() * (255 - umbral_mediana) + umbral_mediana
        transform_brillante = transform_brillante.astype('uint8')

        # Aplicar la transformación a los píxeles de la imagen brillante
        p_brillante = imagen_gris >= umbral_mediana
        imagen_dsihe[p_brillante] = transform_brillante[imagen_gris[p_brillante] - umbral_mediana]

    print("Algoritmo DSIHE aplicado.")
    return imagen_dsihe

def aplicar_bbhe(imagen_original_bgr):
    """
    Aplica el algoritmo de ecualización de bi-histograma que preserva el brillo (BBHE).
    El método segmenta el histograma original en dos partes basándose en la media
    de la imagen, y luego ecualiza cada parte de forma independiente.
    Retorna la imagen mejorada en escala de grises.

    Args:
        imagen_original_bgr (numpy.ndarray): Imagen original en formato BGR (o escala de grises).

    Returns:
        numpy.ndarray: Imagen mejorada con BBHE en escala de grises.
    """
    # Convertir a escala de grises si la imagen es a color
    if len(imagen_original_bgr.shape) == 3:
        imagen_gris = cv2.cvtColor(imagen_original_bgr, cv2.COLOR_BGR2GRAY)
    else:
        imagen_gris = imagen_original_bgr
   
    if imagen_gris.size == 0:
        return imagen_gris

    # Calcular la media (brillo promedio) de la imagen
    media = np.mean(imagen_gris).astype(np.uint8)

    # Crear una imagen vacía del mismo tamaño para el resultado
    imagen_bbhe = np.zeros_like(imagen_gris)

    # Manejar el caso de que la imagen sea muy oscura (media = 0)
    if media == 0:
        media_ajustada = 1
    else:
        media_ajustada = media
   
    # Segmentar la imagen en dos sub-imágenes: una oscura y otra brillante
    imagen_oscura = imagen_gris[imagen_gris < media_ajustada]
    imagen_brillante = imagen_gris[imagen_gris >= media_ajustada]

    # Aplicar ecualización de histograma a cada sub-imagen por separado
    if imagen_oscura.size > 0:
        hist_oscura = cv2.calcHist([imagen_oscura], [0], None, [media_ajustada], [0, media_ajustada]).flatten()
        cdf_oscura = hist_oscura.cumsum()
       
        # Evitar división por cero si el CDF mínimo es igual al máximo
        if cdf_oscura.max() - cdf_oscura.min() > 0:
            cdf_oscura_normalizada = (cdf_oscura - cdf_oscura.min()) / (cdf_oscura.max() - cdf_oscura.min())
            transform_oscura = cdf_oscura_normalizada * (media_ajustada - 1)
        else:
            transform_oscura = np.full_like(cdf_oscura, media_ajustada - 1, dtype=float)
       
        transform_oscura = np.round(transform_oscura)
        transform_oscura = np.clip(transform_oscura, 0, media_ajustada - 1).astype('uint8')
       
        p_oscura = imagen_gris < media_ajustada
        imagen_bbhe[p_oscura] = transform_oscura[imagen_gris[p_oscura]]

    if imagen_brillante.size > 0:
        num_bins_brillante = 255 - media_ajustada
        hist_brillante = cv2.calcHist([imagen_brillante], [0], None, [num_bins_brillante], [media_ajustada, 256]).flatten()
        cdf_brillante = hist_brillante.cumsum()
       
        if cdf_brillante.max() - cdf_brillante.min() > 0:
            cdf_brillante_normalizada = (cdf_brillante - cdf_brillante.min()) / (cdf_brillante.max() - cdf_brillante.min())
            transform_brillante = cdf_brillante_normalizada * (255 - media_ajustada) + media_ajustada
        else:
            transform_brillante = np.full_like(cdf_brillante, 255, dtype=float)
           
        transform_brillante = np.round(transform_brillante)
        transform_brillante = np.clip(transform_brillante, media_ajustada, 255).astype('uint8')

        p_brillante = imagen_gris >= media_ajustada
        mapeo_indices = imagen_gris[p_brillante] - media_ajustada
        imagen_bbhe[p_brillante] = transform_brillante[mapeo_indices]

    print("Algoritmo BBHE aplicado.")
    return imagen_bbhe