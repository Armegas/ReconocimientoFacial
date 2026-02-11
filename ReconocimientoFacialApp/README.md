# Sistema de Reconocimiento Facial

Este proyecto es una aplicación web basada en Flask que permite registrar personas y reconocerlas en tiempo real utilizando la cámara web.

## Características

*   **Registro de Usuarios:** Carga fotos de personas junto con su nombre y número de identificación.
*   **Reconocimiento en Vivo:** Detecta rostros en la transmisión de video y muestra su nombre e ID si están registrados.
*   **Indicador de Desconocido:** Si el rostro no coincide con nadie registrado, se marca como "Desconocido".

## Requisitos

*   Python 3.x
*   Cámara Web

## Instalación

1.  Clona este repositorio o descarga el código.
2.  Instala las dependencias:

    ```bash
    pip install -r requirements.txt
    ```

    *Nota: Se utiliza `opencv-contrib-python` para el reconocedor LBPH.*

## Uso

1.  Ejecuta la aplicación:

    ```bash
    python app.py
    ```

2.  Abre tu navegador y ve a: `http://127.0.0.1:5000`
3.  **Registrar:** Haz clic en "Registrar Nueva Persona", llena el formulario y sube una foto clara del rostro.
4.  **Reconocer:** Vuelve a la página principal para ver el reconocimiento en acción.

## Estructura del Proyecto

*   `app.py`: Archivo principal de la aplicación Flask.
*   `utils.py`: Funciones auxiliares para el entrenamiento del modelo y manejo de imágenes.
*   `templates/`: Archivos HTML para la interfaz.
*   `static/uploads/`: Directorio donde se guardan las fotos subidas.
