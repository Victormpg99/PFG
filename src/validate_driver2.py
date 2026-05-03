import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

try:
    from physics_engine import PhysicsEngine
    from profiler import DriverProfiler
except ImportError:
    print("Error al iniciar la ejecución.")
    exit()

def run_validation():
    print("INICIANDO VALIDACIÓN (Versión Modular)...")
    
    # 1. CARGAR DATOS
    path = r'D:\Universidad\00_PFG\TEST\OBD-II Data_Driver2 - Classified.csv'
    print(f"Leyendo: {path}")
    
    try:
        df = pd.read_csv(path, encoding='ISO-8859-1')
    except:
        df = pd.read_csv(path, encoding='utf-8')

    # 2. ESTANDARIZAR COLUMNAS (Mapeo manual para este dataset específico)
    # Buscamos columnas clave ignorando mayúsculas/minúsculas
    col_speed = next(c for c in df.columns if 'velocidad' in c.lower() and 'km/h' in c.lower())
    col_rpm = next(c for c in df.columns if 'rpm' in c.lower())
    col_label = 'Label' # 0=Agresivo, 1=Moderado

    # Creamos las columnas estándar
    df['speed_kmh'] = pd.to_numeric(df[col_speed], errors='coerce')
    df['rpm'] = pd.to_numeric(df[col_rpm], errors='coerce')
    # Generamos el tiempo (frecuencia de 14.6Hz -> 0.068s)
    df['timestamp_sec'] = np.arange(len(df)) * 0.068

    # 3. APLICAR MOTOR FÍSICO
    print("Calculando física (PhysicsEngine)...")
    physics = PhysicsEngine()
    # Usamos el cálculo automático
    df['accel_ms2'] = physics.calculate_acceleration(df)

    # 4. APLICAR PERFILADO
    print("Clasificando comportamiento (DriverProfiler)...")
    profiler = DriverProfiler()
    # Usamos la lógica centralizada
    df['my_style_detailed'] = df.apply(profiler.classify_moment, axis=1)

    # 5. TRADUCCIÓN (MAPPING) PARA VALIDACIÓN
    # Profiler es detallado (5 clases), el dataset es binario (2 clases).
    # Hacemos la traducción para poder comparar.
    
    def map_to_dataset_format(style):
        # Dataset: 0 = Agresivo, 1 = Moderado
        if style in ['Aggressive_Accel', 'Hard_Braking', 'Inefficient_HighRPM']:
            return 0 # Agresivo
        else:
            return 1 # Moderado (Normal_Eco, Idling)

    df['my_pred_binary'] = df['my_style_detailed'].apply(map_to_dataset_format)

    # 6. MÉTRICAS Y GRÁFICAS
    # Eliminamos filas donde no haya etiqueta real para comparar
    df_val = df.dropna(subset=[col_label, 'my_pred_binary'])
    
    acc = accuracy_score(df_val[col_label], df_val['my_pred_binary'])
    cm = confusion_matrix(df_val[col_label], df_val['my_pred_binary'])

    print(f"\nRESULTADOS DE LA VALIDACIÓN:")
    print(f"   Precisión Global: {acc*100:.2f}%")
    print("\n   Informe de Clasificación:")
    print(classification_report(df_val[col_label], df_val['my_pred_binary'], 
                                target_names=['Agresivo (0)', 'Moderado (1)']))

    # Generar Matriz de Confusión Visual
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred: Agresivo', 'Pred: Moderado'], 
                yticklabels=['Real: Agresivo', 'Real: Moderado'])
    plt.title(f'Matriz de Confusión (Umbral {profiler.HARD_ACCEL_THRESHOLD} m/s²)')
    plt.ylabel('Etiqueta Real (Dataset)')
    plt.xlabel('Predicción (Tu Algoritmo)')
    plt.tight_layout()
    plt.savefig('validacion_modular_matrix.png')
    print("Gráfica guardada: 'validacion_modular_matrix.png'")

if __name__ == "__main__":
    run_validation()