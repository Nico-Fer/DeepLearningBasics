import cv2
from ultralytics import YOLO
import os

# 1. Definir la ruta hacia tu modelo entrenado
# Asumiendo que este script está en python/PotatoSegmentation/server/
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))

# Subimos un nivel para entrar a la carpeta de modelos donde se guardó el entrenamiento
MODEL_PATH = os.path.abspath(os.path.join(DIRECTORIO_ACTUAL, '..', 'models', 'papas_seg', 'weights', 'best.pt'))

def main():
    print(f"Buscando el modelo entrenado en: {MODEL_PATH}")
    
    # Validar que el archivo exista antes de intentar cargarlo
    if not os.path.exists(MODEL_PATH):
        print("Error: No se encontró el archivo best.pt.")
        print("Asegúrate de que el entrenamiento haya terminado correctamente.")
        return

    # 2. Cargar TU modelo
    model = YOLO(MODEL_PATH)

    # 3. Inicializar la cámara web
    # El índice '0' suele ser la cámara principal/integrada de la notebook. 
    # Si tienes una cámara externa por USB, a veces es el '1'.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara web.")
        return

    print("Cámara iniciada. Presiona la tecla 'q' en la ventana de video para salir.")

    # 4. Bucle infinito para leer frame por frame en tiempo real
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al leer la imagen de la cámara.")
            break

        # Hacer la predicción. 
        # conf=0.6 significa que solo mostrará las papas si el modelo está 60% seguro
        results = model.predict(source=frame, conf=0.75, show=False)

        # results[0].plot() toma la imagen original y le dibuja encima la máscara de segmentación
        annotated_frame = results[0].plot()

        # Mostrar el resultado en una ventana de Windows/Mac/Linux
        cv2.imshow('Asistente de Cocina AR - Detección', annotated_frame)

        # 5. Condición de salida (presionar 'q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Limpiar y liberar los recursos de hardware al terminar
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()