import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib # Para guardar/cargar el modelo y ahorrar tiempo

from sklearn.ensemble import RandomForestClassifier

try:
    from data_loader import CarDataStandardizer
except ImportError:
    print("Error inicial")
    exit()

def main():
    print("=== SISTEMA ANALÍTICO ===")
    
    loader = CarDataStandardizer()
    cwd = os.getcwd()

    # --- 1. GESTIÓN DEL MODELO (Carga Inteligente) ---
    model_path = 'robust_model.pkl'
    features = ['speed_kmh', 'rpm', 'accel_ms2', 'jerk_ms3', 
                'accel_std_3s', 'speed_std_3s', 'power_index', 'risk_index']
    
    # Si ya entrenamos el modelo antes, lo cargamos
    if os.path.exists(model_path):
        print(f"\n1️.  Cargando modelo pre-entrenado ('{model_path}')...")
        rf_model = joblib.load(model_path)
    else:
        print("\n1️.  Modelo no encontrado. Iniciando entrenamiento completo...")
        train_df = loader.load_training_dataset(cwd)
        if train_df is None: return
        
        # Data Augmentation (Técnica Robustez)
        X = train_df[features].copy()
        y = train_df['label'].copy()
        X_no_rpm = X.copy()
        X_no_rpm['rpm'] = -1
        
        X_final = pd.concat([X, X_no_rpm])
        y_final = pd.concat([y, y])
        
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, class_weight='balanced')
        rf_model.fit(X_final, y_final)
        joblib.dump(rf_model, model_path)
        print(" Modelo entrenado y guardado.")

    # --- 2. ANÁLISIS DE TU COCHE ---
    print("\n2️. ANALIZANDO TELEMETRÍA...")
    
    # Búsqueda automática del archivo más reciente en TEST o raíz
    import glob
    search_path = os.path.join(cwd, "**", "2026-01-06 12-07-46*.csv") # Busca el archivo
    files = glob.glob(search_path, recursive=True)
    
    if not files:
        print("No encuentro el archivo de hoy (2025-12-29...).")
        # Fallback a manual si quieres
        my_car_path = r"D:\Universidad\00_PFG\PFG_BusCan\TEST\nuevos_datos\2026-01-06 12-07-46.csv"
    else:
        # Cogemos el último modificado
        my_car_path = max(files, key=os.path.getctime)
    
    print(f"Archivo: {os.path.basename(my_car_path)}")
    
    my_df = loader.load_single_file(my_car_path, driver_id=999)
    if my_df is None or my_df.empty: return

    # --- 3. LÓGICA ADAPTATIVA---
    
    # Chequeo de Salud de Sensores
    has_rpm = (my_df['rpm'] > 0).sum() > 10 # ¿Hay datos reales de RPM?
    
    if not has_rpm:
        print(" ALERTA: Datos de motor (RPM) no disponibles o corruptos.")
        print(" Activando Modo 'Cinemática Pura' (Solo GPS/Acelerómetro).")
        print(" Calibrando umbral de sensibilidad a 0.35")
        my_df['rpm'] = -1 # Forzamos el valor que conoce el modelo para "falta de dato"
        threshold = 0.35
        mode_str = "MODO GPS (SENSIBILIDAD ALTA)"
    else:
        print(" Datos de motor (RPM) detectados correctamente.")
        print(" Activando Modo 'Full Telemetry'.")
        threshold = 0.50
        mode_str = "MODO COMPLETO (PRECISIÓN ESTÁNDAR)"

    # Predicción
    probs = rf_model.predict_proba(my_df[features])[:, 0]
    
    # Suavizado inteligente
    my_df['aggression_score'] = pd.Series(probs).rolling(window=75, center=True).mean().fillna(pd.Series(probs))
    # --- LÍMITE DE VELOCIDAD ---
    # Si pasas de 140 km/h, se tiene en cuenta como conducción agresiva el tramo correspondiente.
    HARD_SPEED_LIMIT = 140.0 
    
    # Sobrescribimos el score a 1.0 (Máxima agresividad) donde se supere el límite
    override_mask = my_df['speed_kmh'] > HARD_SPEED_LIMIT
    my_df.loc[override_mask, 'aggression_score'] = 1.0
    
    num_overrides = override_mask.sum()
    if num_overrides > 0:
        print(f" REGLA: Se detectaron {num_overrides} muestras por encima de {HARD_SPEED_LIMIT} km/h.")
    # -----------------------------------------------------

    # Decisión final con el umbral dinámico (0.35 o 0.50)
    my_df['is_aggressive'] = my_df['aggression_score'] > threshold
    
    # Calcular tiempo real en minutos
    my_df['time_min'] = (my_df['timestamp_sec'] - my_df['timestamp_sec'].min()) / 60.0

    # --- 4. INFORME ---
    print("\n=== INFORME DE DIAGNÓSTICO === 📊")
    print(f" Modo de Análisis: {mode_str}")
    
    pct_aggressive = my_df['is_aggressive'].mean() * 100
    duration = my_df['time_min'].max()
    
    print(f"   Duración: {duration:.2f} min")
    print(f"   Nivel de Agresividad: {pct_aggressive:.2f}%")
    
    if pct_aggressive > 40:
        verdict = "CONDUCTOR AGRESIVO"
        color_verdict = 'red'
    elif pct_aggressive > 15:
        verdict = "CONDUCTOR DINÁMICO / NERVIOSO"
        color_verdict = 'orange'
    else:
        verdict = "CONDUCTOR EFICIENTE"
        color_verdict = 'green'

    print(f"CLASIFICACIÓN: [{verdict}]")

    # --- 5. GRÁFICA FINAL ---
    plt.figure(figsize=(14, 8))
    
    # Velocidad y Eventos
    plt.subplot(2, 1, 1)
    plt.plot(my_df['time_min'], my_df['speed_kmh'], color='gray', alpha=0.5, label='Velocidad')
    plt.scatter(my_df[my_df['is_aggressive']]['time_min'], 
                my_df[my_df['is_aggressive']]['speed_kmh'], 
                color='red', s=10, label='Eventos Agresivos', zorder=5)
    plt.title(f"Análisis de Velocidad - {mode_str}")
    plt.ylabel("km/h")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Decisión IA
    plt.subplot(2, 1, 2)
    plt.plot(my_df['time_min'], my_df['aggression_score'], color='purple', label='Probabilidad IA')
    plt.axhline(threshold, color='red', linestyle='--', label=f'Umbral Dinámico ({threshold})')
    plt.fill_between(my_df['time_min'], my_df['aggression_score'], 0, color='purple', alpha=0.1)
    
    # Zona de peligro (para visualizar dónde supera el umbral)
    plt.fill_between(my_df['time_min'], 1, threshold, where=(my_df['aggression_score'] > threshold), 
                     color='red', alpha=0.2, label='Zona Agresiva')
    
    plt.xlabel("Tiempo (Minutos)")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title("Evolución del Comportamiento")
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('INFORME_FINAL_ADAPTATIVO.png')
    print(" Gráfico guardado: 'INFORME_FINAL_ADAPTATIVO.png'")

if __name__ == "__main__":
    main()