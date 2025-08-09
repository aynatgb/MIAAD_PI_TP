import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 

from funciones_mejora import aplicar_ecualizacion_histograma, aplicar_clahe, aplicar_dsihe, aplicar_bbhe
from funciones_metrica import calcular_ambe, calcular_psnr, calcular_contraste, calcular_entropia, mostrar_histograma

if __name__ == "__main__":
    print("--- Ejecutando Ejercicios de Procesamiento de Imágenes ---")

    # --- Configuramos la ruta de la base de datos ---
    RUTA_BASE_DATOS = r'C:\Users\tanya\OneDrive\Escritorio\Procesamiento de imagenes\Trabajo Practico 1\bsds_dataset\BSDS300\images\train'
    
    # Obtenemos la lista completa de imágenes en la base de datos y la ordenamos
    todos_los_archivos = [f for f in os.listdir(RUTA_BASE_DATOS) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    todos_los_archivos = sorted(todos_los_archivos, key=lambda x: int(os.path.splitext(x)[0]))
    
    # --- Solicitamos al usuario el número de imágenes a analizar ---
    num_imagenes_a_analizar = None
    while num_imagenes_a_analizar is None:
        try:
            user_input = input(f"\n¿Cuántas imágenes desea analizar? (Máximo {len(todos_los_archivos)}): ")
            if user_input.lower() == 'todas':
                num_imagenes_a_analizar = len(todos_los_archivos)
            else:
                num_imagenes_a_analizar = int(user_input)
                if not (0 < num_imagenes_a_analizar <= len(todos_los_archivos)):
                    print(f"Por favor, ingrese un número entre 1 y {len(todos_los_archivos)} o 'todas'.")
                    num_imagenes_a_analizar = None
        except ValueError:
            print("Entrada inválida. Por favor, ingrese un número entero o 'todas'.")
            num_imagenes_a_analizar = None

    # Filtramos la lista de imágenes según la entrada del usuario
    IMAGENES_A_PROCESAR = todos_los_archivos[:num_imagenes_a_analizar]
    print(f"Se analizarán {len(IMAGENES_A_PROCESAR)} imágenes.")

    # Listamos para almacenar los resultados de todas las imágenes
    resultados_globales = []

    for nombre_imagen in IMAGENES_A_PROCESAR:
        ruta_completa = os.path.join(RUTA_BASE_DATOS, nombre_imagen)
        imagen_original_bgr = cv2.imread(ruta_completa)

        if imagen_original_bgr is None:
            print(f"Error: No se pudo cargar {nombre_imagen}. Saltando...")
            continue

        print(f"\nProcesando imagen: {nombre_imagen}")
        imagen_original_gris = cv2.cvtColor(imagen_original_bgr, cv2.COLOR_BGR2GRAY)

        # --- Aplicar las cuatro técnicas de Mejora ---
        imagen_he = aplicar_ecualizacion_histograma(imagen_original_bgr)
        imagen_clahe = aplicar_clahe(imagen_original_bgr)
        imagen_dsihe = aplicar_dsihe(imagen_original_bgr)
        imagen_bbhe = aplicar_bbhe(imagen_original_bgr)

        # --- Calculamos las Métricas para cada técnica ---

        metricas_original = {
            "AMBE": np.nan,
            "PSNR": np.nan,
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

        metricas_dsihe = {
            "AMBE": calcular_ambe(imagen_original_gris, imagen_dsihe),
            "PSNR": calcular_psnr(imagen_original_gris, imagen_dsihe),
            "Contraste": calcular_contraste(imagen_dsihe),
            "Entropia": calcular_entropia(imagen_dsihe)
        }
        
        metricas_bbhe = {
            "AMBE": calcular_ambe(imagen_original_gris, imagen_bbhe),
            "PSNR": calcular_psnr(imagen_original_gris, imagen_bbhe),
            "Contraste": calcular_contraste(imagen_bbhe),
            "Entropia": calcular_entropia(imagen_bbhe)
        }

        print("\nMétricas:")
        print(f"Original: {metricas_original}")
        print(f"HE: {metricas_he}")
        print(f"CLAHE: {metricas_clahe}")
        print(f"DSIHE: {metricas_dsihe}")
        print(f"BBHE: {metricas_bbhe}")

        # --- Mostramos una comparación visual de imágenes y sus histogramas ---
        def mostrar_comparacion(original, he, clahe, dsihe, bbhe):
            plt.figure(figsize=(24, 6))

            plt.subplot(1, 5, 1)
            plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            plt.title('Original')
            plt.axis('off')

            plt.subplot(1, 5, 2)
            plt.imshow(he, cmap='gray')
            plt.title('Ecualizacion Histograma (HE)')
            plt.axis('off')

            plt.subplot(1, 5, 3)
            plt.imshow(clahe, cmap='gray')
            plt.title('CLAHE')
            plt.axis('off')
            
            plt.subplot(1, 5, 4)
            plt.imshow(dsihe, cmap='gray')
            plt.title('DSIHE')
            plt.axis('off')
            
            plt.subplot(1, 5, 5)
            plt.imshow(bbhe, cmap='gray')
            plt.title('BBHE')
            plt.axis('off')

            plt.tight_layout()
            plt.show()

        def mostrar_histogramas_comparativos(original_gris, he_gris, clahe_gris, dsihe_gris, bbhe_gris, nombre_img):
            plt.figure(figsize=(24, 5))

            plt.subplot(1, 5, 1)
            plt.hist(original_gris.flatten(), bins=256, range=[0, 256], color='blue', alpha=0.7)
            plt.title(f'Hist. Original ({nombre_img})')
            plt.xlabel("Nivel de Píxel")
            plt.ylabel("Frecuencia")
            plt.grid(True, linestyle='--', alpha=0.6)

            plt.subplot(1, 5, 2)
            plt.hist(he_gris.flatten(), bins=256, range=[0, 256], color='green', alpha=0.7)
            plt.title(f'Hist. HE ({nombre_img})')
            plt.xlabel("Nivel de Píxel")
            plt.grid(True, linestyle='--', alpha=0.6)

            plt.subplot(1, 5, 3)
            plt.hist(clahe_gris.flatten(), bins=256, range=[0, 256], color='red', alpha=0.7)
            plt.title(f'Hist. CLAHE ({nombre_img})')
            plt.xlabel("Nivel de Píxel")
            plt.grid(True, linestyle='--', alpha=0.6)

            plt.subplot(1, 5, 4)
            plt.hist(dsihe_gris.flatten(), bins=256, range=[0, 256], color='purple', alpha=0.7)
            plt.title(f'Hist. DSIHE ({nombre_img})')
            plt.xlabel("Nivel de Píxel")
            plt.grid(True, linestyle='--', alpha=0.6)
            
            plt.subplot(1, 5, 5)
            plt.hist(bbhe_gris.flatten(), bins=256, range=[0, 256], color='orange', alpha=0.7)
            plt.title(f'Hist. BBHE ({nombre_img})')
            plt.xlabel("Nivel de Píxel")
            plt.grid(True, linestyle='--', alpha=0.6)

            plt.tight_layout()
            plt.show()

        mostrar_comparacion(imagen_original_bgr, imagen_he, imagen_clahe, imagen_dsihe, imagen_bbhe)
        mostrar_histogramas_comparativos(imagen_original_gris, imagen_he, imagen_clahe, imagen_dsihe, imagen_bbhe, nombre_imagen)

        # Guardamos los resultados en la lista para generar un cuadro comparativo
        resultados_globales.append({
            "Imagen": nombre_imagen,
            "Técnica": "Original",
            "AMBE": metricas_original["AMBE"],
            "PSNR": metricas_original["PSNR"],
            "Contraste": metricas_original["Contraste"],
            "Entropia": metricas_original["Entropia"]
        })
        resultados_globales.append({
            "Imagen": nombre_imagen,
            "Técnica": "HE",
            "AMBE": metricas_he["AMBE"],
            "PSNR": metricas_he["PSNR"],
            "Contraste": metricas_he["Contraste"],
            "Entropia": metricas_he["Entropia"]
        })
        resultados_globales.append({
            "Imagen": nombre_imagen,
            "Técnica": "CLAHE",
            "AMBE": metricas_clahe["AMBE"],
            "PSNR": metricas_clahe["PSNR"],
            "Contraste": metricas_clahe["Contraste"],
            "Entropia": metricas_clahe["Entropia"]
        })
        resultados_globales.append({
            "Imagen": nombre_imagen,
            "Técnica": "DSIHE",
            "AMBE": metricas_dsihe["AMBE"],
            "PSNR": metricas_dsihe["PSNR"],
            "Contraste": metricas_dsihe["Contraste"],
            "Entropia": metricas_dsihe["Entropia"]
        })
        resultados_globales.append({
            "Imagen": nombre_imagen,
            "Técnica": "BBHE",
            "AMBE": metricas_bbhe["AMBE"],
            "PSNR": metricas_bbhe["PSNR"],
            "Contraste": metricas_bbhe["Contraste"],
            "Entropia": metricas_bbhe["Entropia"]
        })

    print("\n--- Análisis Completado ---")

    # --- Mostramos la tabla comparativa ---
    if len(IMAGENES_A_PROCESAR) > 0:
        print("\n--- Cuadro Comparativo de Métricas (Detallado por Imagen) ---")
        df = pd.DataFrame(resultados_globales)
        df_pivot = df.pivot_table(index=['Imagen', 'Técnica'], values=['AMBE', 'PSNR', 'Contraste', 'Entropia'])
        print(df_pivot)
        print("\n")
    else:
        print("\nNo se analizó ninguna imagen.")
