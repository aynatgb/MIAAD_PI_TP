# 🎨 Análisis Comparativo de Técnicas de Mejora de Imagen


## 🚀 _Estructura del Proyecto_

* **`main_proyecto.py`**: Es el script de orquestación principal. Su función es cargar las imágenes, aplicar las técnicas de mejora, calcular las métricas de rendimiento y generar visualizaciones.
* **`funciones_mejora.py`**: Este módulo contiene las implementaciones de los cuatro algoritmos de mejora de imagen (**HE**, **CLAHE**, **DSIHE**, **BBHE**).
* **`funciones_metrica.py`**: Contiene las funciones para calcular las métricas de evaluación (**AMBE**, **PSNR**, **Contraste**, **Entropía**) y para visualizar los histogramas.
* **`README.md`**: Este archivo proporciona una guía completa sobre la configuración, ejecución y consideraciones del proyecto.

---

## 🛠️ _Requisitos del Entorno_

Para replicar y ejecutar el análisis, se requiere un entorno Python con las siguientes librerías instaladas:

* **OpenCV** (`opencv-python`): Para operaciones de procesamiento de imágenes.
* **NumPy** (`numpy`): Esencial para la manipulación eficiente de arrays de píxeles.
* **Matplotlib** (`matplotlib`): Utilizado para la generación de gráficos.
* **Pandas** (`pandas`): Empleado para la organización y manipulación de datos.

Se puede instalar todas las librerias necesarias de una sola vez mediante el siguiente comando:

```bash
pip install opencv-python numpy matplotlib pandas 

```

## ⚙️ _Configuración y Uso_

**Base de Datos de Imágenes:** El proyecto está configurado para utilizar una base de datos de imágenes. Por defecto, la ruta es:
```
C:\\\\Users\\\\tanya\\\\OneDrive\\\\Escritorio\\\\Procesamiento de imagenes\\\\Trabajo Practico 1\\\\bsds\_dataset\\\\BSDS300\\\\images\\\\train
```

Asegúrate de cambiar la variable **RUTA\_BASE\_DATOS en main\_proyecto.py** para que apunte a la ubicación correcta de tu base de datos de imágenes.



**Ejecución:** Para iniciar el análisis, simplemente ejecuta el script principal desde la terminal:


```
python main\_proyecto.py
```


**Selección de Imágenes:** El programa te pedirá que ingreses la cantidad de imágenes que deseas analizar. Puedes ingresar un número o la palabra todas para procesar el conjunto completo.



# 📝 _Consideraciones Adicionales_

**Entrada y Salida:** El programa mostrará los resultados de las métricas en la consola y generará gráficos para la comparación visual de las imágenes y sus histogramas.



**Ventanas de Visualización:** Al ejecutar el script, se abrirán ventanas de visualización de matplotlib y OpenCV. Deberás cerrarlas manualmente para que el programa continúe su ejecución o finalice.



**Manejo de Errores:** El código incluye manejo de errores básico para la carga de imágenes. Si una imagen no se puede cargar, el programa saltará a la siguiente.


