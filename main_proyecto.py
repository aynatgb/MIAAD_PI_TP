# main_proyecto.py
import cv2
import os
import matplotlib.pyplot as plt # Para la visualización de comparación

# Importa las funciones de mejora y métricas que has definido
# Asegúrate de que estos archivos (funciones_mejora.py y funciones_metrica.py)
# estén en la misma carpeta o en una ruta accesible.
from funciones_mejora import aplicar_ecualizacion_histograma, aplicar_clahe # Asume que estas funciones existen
from funciones_metrica import calcular_ambe, calcular_psnr, calcular_contraste, calcular_entropia, mostrar_histograma

# --- Configuración de la Base de Datos ---
# ¡CAMBIA ESTA RUTA A LA UBICACIÓN REAL DE TUS IMÁGENES BSDS!
# Por ejemplo, si descargaste las imágenes de entrenamiento en 'BSDS300/images/train/'
RUTA_BASE_DATOS = r'C:\Users\tanya\OneDrive\Escritorio\Procesamiento de imagenes\Trabajo Practico 1\bsds_dataset\BSDS300\images\train'

# Puedes listar algunas imágenes específicas o iterar sobre todas las de una carpeta
# Para empezar, puedes tomar algunas imágenes de ejemplo de la carpeta 'train' o 'test'
# IMAGENES_A_PROCESAR = ['imagen_ejemplo_1.jpg', 'imagen_ejemplo_2.jpg']
# O para procesar todas las imágenes en la carpeta:
IMAGENES_A_PROCESAR = [f for f in os.listdir(RUTA_BASE_DATOS) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]


if __name__ == "__main__":
    resultados_globales = {}

    for nombre_imagen in IMAGENES_A_PROCESAR:
        ruta_completa = os.path.join(RUTA_BASE_DATOS, nombre_imagen)
        imagen_original_bgr = cv2.imread(ruta_completa)

        if imagen_original_bgr is None:
            print(f"Error: No se pudo cargar {nombre_imagen}. Saltando...")
            continue

        print(f"\nProcesando imagen: {nombre_imagen}")

        # Convertir a gris para métricas y HE/CLAHE
        imagen_original_gris = cv2.cvtColor(imagen_original_bgr, cv2.COLOR_BGR2GRAY)

        # --- Aplicar Técnicas de Mejora ---
        # Estas funciones deben estar definidas en funciones_mejora.py
        imagen_he = aplicar_ecualizacion_histograma(imagen_original_bgr) # Retorna gris
        imagen_clahe = aplicar_clahe(imagen_original_bgr) # Retorna gris
        
        # --- Implementa tu algoritmo adicional aquí ---
        # Por ejemplo, un placeholder:
        # imagen_adicional = tu_algoritmo_adicional(imagen_original_bgr) # Asegúrate de que retorne gris o BGR
        # Si no tienes un algoritmo adicional todavía, puedes usar una copia de la original en gris
        imagen_adicional = imagen_original_gris.copy() # Placeholder
        titulo_adicional = "Algoritmo Adicional (Placeholder)"


        # --- Calcular Métricas ---
        # Asegúrate de que las imágenes para las métricas estén en escala de grises
        # Si tu algoritmo adicional devuelve color, conviértelo a gris para las métricas
        if len(imagen_adicional.shape) == 3:
            imagen_adicional_gris = cv2.cvtColor(imagen_adicional, cv2.COLOR_BGR2GRAY)
        else:
            imagen_adicional_gris = imagen_adicional

        metricas_original = {
            "AMBE": "-", # No aplica para original vs original
            "PSNR": "-", # No aplica para original vs original
            "Contraste": calcular_contraste(imagen_original_gris),
            "Entropia": calcular_entropia(imagen_original_gris)
        }

        metricas_he = {
            "AMBE": calcular_ambe(imagen_original_gris, imagen_he),
            "PSNR": calcular_psnr(imagen_original_gris, imagen_he),
            "Contraste": calcular_contraste(imagen_he),
            "Entropia": calcular_entropia(imagen_he)
        }

        metricas_clahe = {
            "AMBE": calcular_ambe(imagen_original_gris, imagen_clahe),
            "PSNR": calcular_psnr(imagen_original_gris, imagen_clahe),
            "Contraste": calcular_contraste(imagen_clahe),
            "Entropia": calcular_entropia(imagen_clahe)
        }

        metricas_adicional = {
            "AMBE": calcular_ambe(imagen_original_gris, imagen_adicional_gris),
            "PSNR": calcular_psnr(imagen_original_gris, imagen_adicional_gris),
            "Contraste": calcular_contraste(imagen_adicional_gris),
            "Entropia": calcular_entropia(imagen_adicional_gris)
        }

        print("\nMétricas:")
        print(f"Original: {metricas_original}")
        print(f"HE: {metricas_he}")
        print(f"CLAHE: {metricas_clahe}")
        print(f"{titulo_adicional}: {metricas_adicional}")

        # --- Mostrar Comparación Visual de Imágenes ---
        # Puedes definir esta función en funciones_visualizacion.py o directamente aquí
        def mostrar_comparacion(original, he, clahe, adicional_img, titulo_adicional_plot="Algoritmo Adicional"):
            plt.figure(figsize=(18, 6))

            plt.subplot(1, 4, 1)
            plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB)) # Convertir a RGB para matplotlib
            plt.title('Original')
            plt.axis('off')

            plt.subplot(1, 4, 2)
            plt.imshow(he, cmap='gray')
            plt.title('Ecualizacion Histograma (HE)')
            plt.axis('off')

            plt.subplot(1, 4, 3)
            plt.imshow(clahe, cmap='gray')
            plt.title('CLAHE')
            plt.axis('off')

            plt.subplot(1, 4, 4)
            # Asumimos que el algoritmo adicional devuelve BGR si es a color, o gris si es gris
            if len(adicional_img.shape) == 3:
                plt.imshow(cv2.cvtColor(adicional_img, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(adicional_img, cmap='gray')
            plt.title(titulo_adicional_plot)
            plt.axis('off')

            plt.tight_layout()
            plt.show()

        mostrar_comparacion(imagen_original_bgr, imagen_he, imagen_clahe, imagen_adicional, titulo_adicional)

        # --- Mostrar Histogramas Comparativos ---
        # Puedes definir esta función en funciones_visualizacion.py o directamente aquí
        def mostrar_histogramas_comparativos(original_gris, he_gris, clahe_gris, adicional_gris, nombre_img):
            plt.figure(figsize=(18, 5))

            plt.subplot(1, 4, 1)
            plt.hist(original_gris.flatten(), bins=256, range=[0, 256], color='blue', alpha=0.7)
            plt.title(f'Hist. Original ({nombre_img})')
            plt.xlabel("Nivel de Píxel")
            plt.ylabel("Frecuencia")
            plt.grid(True, linestyle='--', alpha=0.6)

            plt.subplot(1, 4, 2)
            plt.hist(he_gris.flatten(), bins=256, range=[0, 256], color='green', alpha=0.7)
            plt.title(f'Hist. HE ({nombre_img})')
            plt.xlabel("Nivel de Píxel")
            plt.grid(True, linestyle='--', alpha=0.6)

            plt.subplot(1, 4, 3)
            plt.hist(clahe_gris.flatten(), bins=256, range=[0, 256], color='red', alpha=0.7)
            plt.title(f'Hist. CLAHE ({nombre_img})')
            plt.xlabel("Nivel de Píxel")
            plt.grid(True, linestyle='--', alpha=0.6)

            plt.subplot(1, 4, 4)
            plt.hist(adicional_gris.flatten(), bins=256, range=[0, 256], color='purple', alpha=0.7)
            plt.title(f'Hist. Adicional ({nombre_img})')
            plt.xlabel("Nivel de Píxel")
            plt.grid(True, linestyle='--', alpha=0.6)

            plt.tight_layout()
            plt.show()

        mostrar_histogramas_comparativos(imagen_original_gris, imagen_he, imagen_clahe, imagen_adicional_gris, nombre_imagen)


        # Guardar resultados (opcional)
        resultados_globales[nombre_imagen] = {
            "original": metricas_original,
            "HE": metricas_he,
            "CLAHE": metricas_clahe,
            "Adicional": metricas_adicional
        }

    print("\n--- Análisis Completado ---")
    # Puedes imprimir resultados_globales o guardarlos en un archivo CSV/JSON
    # import json
    # with open('resultados_analisis.json', 'w') as f:
    #     json.dump(resultados_globales, f, indent=4)
