from ultralytics import YOLO
import os

# 1. Definir rutas dinámicas adaptadas a tu estructura
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__)) # Apunta a python/PotatoSegmentation/training/

# Subimos 3 niveles usando '..' para llegar a la raíz (donde conviven DataSets y python)
DIRECTORIO_RAIZ = os.path.abspath(os.path.join(DIRECTORIO_ACTUAL, '..', '..', '..'))

# Bajamos hacia el yaml (respetando tus mayúsculas exactas)
YAML_PATH = os.path.join(DIRECTORIO_RAIZ, 'DataSets', 'PotatoSegmentation', 'Data.yaml')

def main():
    print("Iniciando la carga del modelo...")
    # 2. Cargar el modelo base pre-entrenado
    # La primera vez que corras esto, descargará el archivo .pt automáticamente
    model = YOLO('yolo26m-seg.pt') 

    print(f"Comenzando entrenamiento con dataset en: {YAML_PATH}")
    # 3. Entrenar el modelo
    results = model.train(
        data=YAML_PATH,
        epochs=50,               # Cantidad de veces que el modelo verá el dataset completo
        imgsz=640,               # Tamaño al que se redimensionarán las imágenes (estándar de YOLO)
        batch=16,                # Cuántas imágenes procesa a la vez. (Si te da error de memoria en la gráfica, bajalo a 8 o 4)
        project=os.path.join(DIRECTORIO_RAIZ, 'python', 'PotatoSegmentation', 'models'),
        name='papas_seg',        # Nombre de la carpeta específica de este entrenamiento
        device=0                 # '0' usa tu tarjeta gráfica NVIDIA. Si no tienes o falla, cámbialo a 'cpu'
    )
    
    print("¡Entrenamiento finalizado!")

if __name__ == '__main__':
    main()