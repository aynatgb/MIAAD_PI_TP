import cv2
import numpy as np

def aplicar_ecualizacion_histograma(imagen_bgr):
    """
    Aplicamos la ecualización de histograma estándar (HE) a la imagen.
    Si la imagen ya está en escala de grises, se usa directamente.
    Si no, se convierte a escala de grises antes de aplicar la ecualización.
    """
    # Verificamos si la imagen tiene 3 canales (color) o 1 (escala de grises)
    # Se usa el nombre de variable original 'imagen_bgr' para la entrada,
    # pero el código ahora maneja ambos casos.
    if len(imagen_bgr.shape) == 3:
        # Si es a color, la convertimos a escala de grises
        imagen_gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
        print("La imagen fue convertida a escala de grises.")
    else:
        # Si ya está en escala de grises, la usamos directamente
        imagen_gris = imagen_bgr
        print("La imagen ya está en escala de grises, no se requiere conversión.")

    imagen_ecualizada = cv2.equalizeHist(imagen_gris)
    return imagen_ecualizada

def aplicar_clahe(imagen_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Aplicamos la ecualización de histograma adaptativa (CLAHE) a la imagen.
    """
    imagen_gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    imagen_clahe = clahe.apply(imagen_gris)
    return imagen_clahe

def aplicar_dsihe(imagen_bgr):
    """
    Aplicamos la técnica de Ecualización de Histograma Sub-imagen Dinámica (DSIHE).
    """
    print("Algoritmo DSIHE aplicado.")
    imagen_gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    
    # Calculamos el histograma de la imagen
    hist = cv2.calcHist([imagen_gris], [0], None, [256], [0, 256])

    # Encontramos el valor medio de los píxeles de la imagen
    media_intensidad = np.mean(imagen_gris)

    # Dividimos el histograma en dos partes
    hist_izq = hist[0:int(media_intensidad) + 1]
    hist_der = hist[int(media_intensidad) + 1:256]

    # Calculamos las probabilidades acumuladas (CDF) para cada sub-histograma
    cdf_izq = np.cumsum(hist_izq)
    cdf_der = np.cumsum(hist_der)

    cdf_izq_max = cdf_izq.max()
    cdf_der_max = cdf_der.max()
    cdf_izq_norm = cdf_izq / cdf_izq_max if cdf_izq_max > 0 else np.zeros_like(cdf_izq)
    cdf_der_norm = cdf_der / cdf_der_max if cdf_der_max > 0 else np.zeros_like(cdf_der)

    # Calculamos las transformaciones de píxeles
    transform_izq = np.round(media_intensidad * cdf_izq_norm).astype('uint8')
    transform_der = np.round(255 - media_intensidad + (media_intensidad * cdf_der_norm)).astype('uint8')
    transform_der += int(media_intensidad)

    # Aplicamos las transformaciones a los píxeles de la imagen
    imagen_dsihe = np.zeros_like(imagen_gris)
    imagen_dsihe[imagen_gris <= media_intensidad] = transform_izq[imagen_gris[imagen_gris <= media_intensidad]]
    imagen_dsihe[imagen_gris > media_intensidad] = transform_der[imagen_gris[imagen_gris > media_intensidad] - int(media_intensidad) - 1]

    return imagen_dsihe

def aplicar_bbhe(imagen_bgr):
    """
    Aplica la técnica de Ecualización de Histograma por Separación de Brillo (BBHE).
    """
    print("Algoritmo BBHE aplicado.")
    imagen_gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    
    # Calculamos la media de los píxeles de la imagen
    media_intensidad = np.mean(imagen_gris)
    
    # Separamos la imagen en píxeles oscuros y brillantes
    p_oscuro = imagen_gris < media_intensidad
    p_brillante = imagen_gris >= media_intensidad

    # Creamos las sub-imágenes
    imagen_oscura = np.copy(imagen_gris)
    imagen_oscura[p_brillante] = 0
    imagen_brillante = np.copy(imagen_gris)
    imagen_brillante[p_oscuro] = 0
    
    # Aplicamos ecualización a los sub-histogramas
    hist_oscuro, _ = np.histogram(imagen_oscura[p_oscuro], 256, [0, 256])
    cdf_oscuro = hist_oscuro.cumsum()
    # Agregamos una comprobación para evitar la división por cero
    cdf_oscuro_max = cdf_oscuro.max()
    cdf_oscuro_norm = cdf_oscuro / cdf_oscuro_max if cdf_oscuro_max > 0 else np.zeros_like(cdf_oscuro)
    transform_oscuro = np.round(media_intensidad * cdf_oscuro_norm).astype('uint8')
    
    # Sub-imagen brillante
    hist_brillante, _ = np.histogram(imagen_brillante[p_brillante], 256, [0, 256])
    cdf_brillante = hist_brillante.cumsum()
    # Agregamos una comprobación para evitar la división por cero
    cdf_brillante_max = cdf_brillante.max()
    cdf_brillante_norm = cdf_brillante / cdf_brillante_max if cdf_brillante_max > 0 else np.zeros_like(cdf_brillante)
    transform_brillante = np.round((255 - media_intensidad) * cdf_brillante_norm).astype('uint8')
    transform_brillante += int(media_intensidad)
    
    # Combinamos los resultados
    imagen_bbhe = np.zeros_like(imagen_gris)
    
    # Corregimos la asignación de píxeles
    imagen_bbhe[p_oscuro] = transform_oscuro[imagen_gris[p_oscuro]]
    imagen_bbhe[p_brillante] = transform_brillante[imagen_gris[p_brillante] - int(media_intensidad)]
    
    return imagen_bbhe

