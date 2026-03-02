import cv2
import numpy as np
import math
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

        annotated_frame = frame.copy()  # Copia del frame original para dibujar las anotaciones

        # Verificar si el modelo detectó algo y si generó máscaras
        if len(results) > 0 and results[0].masks is not None:
            
            # Extraer solo los contornos (ignoramos las cajas rectas clásicas)
            contornos = results[0].masks.xy             

            # Iterar sobre cada papa detectada
            for i in range(len(contornos)):
                # Convertir el contorno a formato numpy para que OpenCV lo entienda
                contorno_actual = np.array(contornos[i], dtype=np.float32)
                
                # Filtro de seguridad: ignorar contornos diminutos (ruido o fallos de 1 píxel)
                if len(contorno_actual) < 5:
                    continue

                # 1. Obtener el rectángulo rotado que envuelve la máscara
                # 'rect' devuelve: (centro(x,y), (ancho, alto), ángulo de rotación)
                rect = cv2.minAreaRect(contorno_actual)
                
                # 2. Extraer las coordenadas exactas de las 4 esquinas de ese rectángulo
                box = cv2.boxPoints(rect)
                box = np.int32(box) # Convertir a números enteros (píxeles)

                # Dibujar la silueta real de la papa en azul para referencia visual
                contorno_int = np.array(contornos[i], dtype=np.int32)
                cv2.polylines(annotated_frame, [contorno_int], isClosed=True, color=(255, 0, 0), thickness=2)

                # 3. Matemática para el corte longitudinal
                # Nombramos las 4 esquinas del rectángulo
                pt0, pt1, pt2, pt3 = box
                
                # Calculamos la longitud de dos lados adyacentes usando el Teorema de Pitágoras
                dist_0_1 = math.hypot(pt0[0] - pt1[0], pt0[1] - pt1[1])
                dist_1_2 = math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

                # Buscamos los lados más cortos y unimos sus puntos medios
                if dist_0_1 < dist_1_2:
                    # El lado 0-1 es el ancho (corto). Unimos su centro con el opuesto (2-3)
                    punto_A = (int((pt0[0] + pt1[0]) / 2), int((pt0[1] + pt1[1]) / 2))
                    punto_B = (int((pt2[0] + pt3[0]) / 2), int((pt2[1] + pt3[1]) / 2))
                else:
                    # El lado 1-2 es el ancho (corto). Unimos su centro con el opuesto (3-0)
                    punto_A = (int((pt1[0] + pt2[0]) / 2), int((pt1[1] + pt2[1]) / 2))
                    punto_B = (int((pt3[0] + pt0[0]) / 2), int((pt3[1] + pt0[1]) / 2))

                # 4. LA ZONA DE CORTE DINÁMICA: Trazar la línea
                cv2.line(annotated_frame, punto_A, punto_B, color=(0, 255, 0), thickness=3)
                
                # 5. Agregar el texto de guía en el centro exacto de la papa
                centro_x, centro_y = int(rect[0][0]), int(rect[0][1])
                cv2.putText(annotated_frame, "Cortar aqui", (centro_x - 40, centro_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Asistente de Cocina AR - Tesis', annotated_frame)
        
        # 5. Condición de salida (presionar 'q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Limpiar y liberar los recursos de hardware al terminar
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()