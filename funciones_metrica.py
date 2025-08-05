import numpy as np
import cv2
import scipy.stats
import matplotlib.pyplot as plt # Importado para gráficos

def calcular_ambe(imagen_original_gris, imagen_mejorada_gris):
    """
    Calcula el Absolute Mean Brightness Error (AMBE).
    Mide la diferencia absoluta entre el brillo medio de la imagen original
    y la imagen mejorada. Un valor más cercano a cero indica que la técnica
    no ha alterado drásticamente el brillo general.

    Args:
        imagen_original_gris (numpy.ndarray): Imagen original en escala de grises.
        imagen_mejorada_gris (numpy.ndarray): Imagen mejorada en escala de grises.

    Returns:
        float: El valor de AMBE.
    """
    if imagen_original_gris.shape != imagen_mejorada_gris.shape:
        raise ValueError("Las imágenes deben tener las mismas dimensiones para calcular AMBE.")
    
    brillo_original = np.mean(imagen_original_gris)
    brillo_mejorado = np.mean(imagen_mejorada_gris)
    ambe = abs(brillo_mejorado - brillo_original)
    return ambe

def calcular_psnr(imagen_original_gris, imagen_mejorada_gris):
    """
    Calcula el Peak Signal-to-Noise Ratio (PSNR).
    Mide la relación entre la potencia máxima de una señal y la potencia del ruido.
    Un PSNR más alto generalmente indica una mejor calidad de reconstrucción.

    Args:
        imagen_original_gris (numpy.ndarray): Imagen original en escala de grises.
        imagen_mejorada_gris (numpy.ndarray): Imagen mejorada en escala de grises.

    Returns:
        float: El valor de PSNR. Retorna np.inf si las imágenes son idénticas.
    """
    if imagen_original_gris.shape != imagen_mejorada_gris.shape:
        raise ValueError("Las imágenes deben tener las mismas dimensiones para calcular PSNR.")
    
    # cv2.PSNR ya espera arrays NumPy y maneja el cálculo directamente.
    # Asegúrate de que las imágenes sean del mismo tipo de dato y tamaño.
    psnr = cv2.PSNR(imagen_original_gris, imagen_mejorada_gris)
    return psnr

def calcular_contraste(imagen_gris):
    """
    Calcula el contraste de una imagen en escala de grises como su desviación estándar.
    Un valor más alto indica un mayor contraste.

    Args:
        imagen_gris (numpy.ndarray): Imagen en escala de grises.

    Returns:
        float: El valor de contraste (desviación estándar).
    """
    if imagen_gris.size == 0: # Evitar error si la imagen está vacía
        return 0.0
    contraste = np.std(imagen_gris)
    return contraste

def calcular_entropia(imagen_gris):
    """
    Calcula la entropía de una imagen en escala de grises utilizando la fórmula de Shannon.
    La entropía mide la cantidad de información o la aleatoriedad en una imagen.
    Una imagen con más detalles y una distribución de píxeles más variada tendrá una entropía más alta.

    Args:
        imagen_gris (numpy.ndarray): Imagen en escala de grises.

    Returns:
        float: El valor de entropía.
    """
    if imagen_gris.size == 0: # Evitar error si la imagen está vacía
        return 0.0
        
    # Calcular el histograma de la imagen (frecuencia de cada nivel de gris)
    # cv2.calcHist devuelve un array de 256x1, lo aplanamos.
    hist = cv2.calcHist([imagen_gris], [0], None, [256], [0, 256]).flatten()

    # Normalizar el histograma para obtener las probabilidades de cada nivel de gris
    # Suma total de píxeles para evitar división por cero si la imagen es vacía
    total_pixels = imagen_gris.size
    if total_pixels == 0:
        return 0.0
    
    hist_normalized = hist / total_pixels

    # Calcular la entropía (usando la fórmula de Shannon)
    # Ignorar los bins con probabilidad cero para evitar log2(0)
    # np.log2(p + 1e-8) se usa para evitar log de cero, añadiendo un pequeño valor
    entropia = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-8))
    return entropia

def mostrar_histograma(imagen_gris, titulo="Histograma"):
    """
    Muestra el histograma de una imagen en escala de grises.

    Args:
        imagen_gris (numpy.ndarray): Imagen en escala de grises.
        titulo (str): Título del gráfico del histograma.
    """
    plt.figure(figsize=(6, 4))
    plt.hist(imagen_gris.flatten(), bins=256, range=[0, 256], color='gray', alpha=0.7)
    plt.title(titulo)
    plt.xlabel("Nivel de Píxel")
    plt.ylabel("Frecuencia")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# Ejemplo de uso (opcional, para probar las funciones de métrica directamente)
if __name__ == "__main__":
    # Crear imágenes de ejemplo (en escala de grises)
    img_original = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
    img_mejorada = np.array([[20, 30, 40], [50, 60, 70], [80, 90, 100]], dtype=np.uint8)
    img_homogenea = np.full((3, 3), 128, dtype=np.uint8) # Imagen con bajo contraste
    img_ruido = np.random.randint(0, 256, size=(3, 3), dtype=np.uint8) # Imagen con alto contraste/ruido

    print("--- Pruebas de Métricas ---")
    print(f"Imagen Original:\n{img_original}")
    print(f"Imagen Mejorada:\n{img_mejorada}")
    print(f"Imagen Homogénea:\n{img_homogenea}")
    print(f"Imagen con Ruido:\n{img_ruido}")

    # AMBE
    ambe_val = calcular_ambe(img_original, img_mejorada)
    print(f"\nAMBE (Original vs Mejorada): {ambe_val:.4f}")

    # PSNR
    psnr_val = calcular_psnr(img_original, img_mejorada)
    print(f"PSNR (Original vs Mejorada): {psnr_val:.4f}")

    # Contraste
    contraste_original = calcular_contraste(img_original)
    contraste_mejorada = calcular_contraste(img_mejorada)
    contraste_homogenea = calcular_contraste(img_homogenea)
    contraste_ruido = calcular_contraste(img_ruido)
    print(f"\nContraste (Original): {contraste_original:.4f}")
    print(f"Contraste (Mejorada): {contraste_mejorada:.4f}")
    print(f"Contraste (Homogénea): {contraste_homogenea:.4f}")
    print(f"Contraste (Ruido): {contraste_ruido:.4f}")

    # Entropía
    entropia_original = calcular_entropia(img_original)
    entropia_mejorada = calcular_entropia(img_mejorada)
    entropia_homogenea = calcular_entropia(img_homogenea)
    entropia_ruido = calcular_entropia(img_ruido)
    print(f"\nEntropía (Original): {entropia_original:.4f}")
    print(f"Entropía (Mejorada): {entropia_mejorada:.4f}")
    print(f"Entropía (Homogénea): {entropia_homogenea:.4f}")
    print(f"Entropía (Ruido): {entropia_ruido:.4f}")

    # --- Visualización de Imágenes y Histogramas ---
    print("\n--- Visualizando Imágenes y Histogramas ---")

    # Mostrar imágenes de ejemplo
    cv2.imshow("Imagen Original", img_original)
    cv2.imshow("Imagen Mejorada", img_mejorada)
    cv2.imshow("Imagen Homogenea", img_homogenea)
    cv2.imshow("Imagen con Ruido", img_ruido)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Mostrar histogramas de las imágenes de ejemplo
    mostrar_histograma(img_original, "Histograma - Imagen Original")
    mostrar_histograma(img_mejorada, "Histograma - Imagen Mejorada")
    mostrar_histograma(img_homogenea, "Histograma - Imagen Homogenea")
    mostrar_histograma(img_ruido, "Histograma - Imagen con Ruido")
