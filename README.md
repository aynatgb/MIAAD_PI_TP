**Estructura del Proyecto**

El proyecto está organizado en tres archivos principales:

main\_proyecto.py: El script principal que coordina la ejecución. Carga las imágenes, aplica cada una de las técnicas de mejora, calcula las métricas de evaluación y muestra los resultados.

funciones\_mejora.py: Contiene las implementaciones de los cuatro algoritmos de mejora de imagen (HE, CLAHE, DSIHE, BBHE).

funciones\_metrica.py: Contiene las funciones para calcular las métricas de evaluación (AMBE, PSNR, Contraste, Entropía) y para visualizar los histogramas.



**Requisitos**

Para ejecutar este proyecto, necesitas tener instaladas las siguientes librerías de Python:

OpenCV (cv2)

NumPy (numpy)

Matplotlib (matplotlib)

Pandas (pandas)





**Puedes instalar todas las dependencias con pip:**

pip install opencv-python numpy matplotlib pandas



**Configuración y Uso**

Base de Datos de Imágenes: El proyecto está configurado para utilizar una base de datos de imágenes. Por defecto, la ruta es:

C:\\\\Users\\\\tanya\\\\OneDrive\\\\Escritorio\\\\Procesamiento de imagenes\\\\Trabajo Practico 1\\\\bsds\_dataset\\\\BSDS300\\\\images\\\\train



Asegúrate de cambiar la variable RUTA\_BASE\_DATOS en main\_proyecto.py para que apunte a la ubicación correcta de tu base de datos de imágenes.



**Ejecución:** Para iniciar el análisis, simplemente ejecuta el script principal desde la terminal:



python main\_proyecto.py



**Selección de Imágenes:** El programa te pedirá que ingreses la cantidad de imágenes que deseas analizar. Puedes ingresar un número o la palabra todas para procesar el conjunto completo.



**Consideraciones Adicionales**

**Entrada y Salida:** El programa mostrará los resultados de las métricas en la consola y generará gráficos para la comparación visual de las imágenes y sus histogramas.



**Ventanas de Visualización:** Al ejecutar el script, se abrirán ventanas de visualización de matplotlib y OpenCV. Deberás cerrarlas manualmente para que el programa continúe su ejecución o finalice.



**Manejo de Errores:** El código incluye manejo de errores básico para la carga de imágenes. Si una imagen no se puede cargar, el programa saltará a la siguiente.

