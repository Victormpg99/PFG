import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns

try:
    from data_loader import CarDataStandardizer
except ImportError:
    print("Error inicial")
    exit()

def main():
    print("INICIANDO ENTRENAMIENTO DEL MODELO (v3.1)...")
    loader = CarDataStandardizer()
    cwd = os.getcwd()
    
    # 1. CARGA DE DATOS
    print("Buscando datasets etiquetados...")
    train_df = loader.load_training_dataset(cwd)
    
    if train_df is None:
        test_path = os.path.join(cwd, "TEST")
        if os.path.exists(test_path):
            train_df = loader.load_training_dataset(test_path)
    
    if train_df is None or train_df.empty:
        print("No se encontraron datos de entrenamiento.")
        return

    print(f"Datos brutos cargados: {len(train_df)} muestras.")

    # Eliminamos filas donde la etiqueta sea NaN (vacía o corrupta)
    # y aseguramos que sean enteros (0 o 1).
    initial_len = len(train_df)
    train_df = train_df.dropna(subset=['label'])
    dropped = initial_len - len(train_df)
    
    if dropped > 0:
        print(f"Se eliminaron {dropped} filas con etiquetas corruptas (NaN).")
    
    train_df['label'] = train_df['label'].astype(int)
    # -----------------------------------------------

    # 2. ESTUDIO DE CARACTERÍSTICAS
    features = ['speed_kmh', 'rpm', 'accel_ms2', 'jerk_ms3', 
                'accel_std_3s', 'speed_std_3s', 'power_index', 'risk_index']
    
    for f in features:
        if f not in train_df.columns: train_df[f] = 0

    # 3. DATA AUGMENTATION
    print("Aplicando Data Augmentation (Simulación de fallo OBD)...")
    aug_df = train_df.copy()
    aug_df['rpm'] = -1 
    aug_df = aug_df.sample(frac=0.5, random_state=42)
    
    final_train_df = pd.concat([train_df, aug_df], ignore_index=True)
    
    # Comprobación final antes de entrenar
    if final_train_df['label'].isna().any():
        print("Aún quedan NaNs en las etiquetas tras el aumento.")
        return

    X = final_train_df[features]
    y = final_train_df['label']

    # 4. ENTRENAMIENTO
    print(f"Entrenando Random Forest con {len(X)} muestras...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, class_weight='balanced')
    
    try:
        rf_model.fit(X, y)
    except ValueError as e:
        print(f" Error durante el ajuste (fit): {e}")
        # Debugging en entrenamiento
        print("DEBUG X NaN check:", X.isna().sum())
        print("DEBUG y NaN check:", y.isna().sum())
        return

    # 5. GUARDADO
    joblib.dump(rf_model, 'robust_model.pkl')
    print("Modelo guardado como 'robust_model.pkl'")

    # 6. AUTO-VALIDACIÓN
    print("\n--- VALIDACIÓN DEL MODELO ---")
    
    my_df = train_df.copy() # Validamos sobre datos reales (no aumentados)
    
    # Manejo de clases (asegurar cuál es agresivo)
    probs = rf_model.predict_proba(my_df[features])
    # Si classes_ es [0, 1] y Label 0 es Agresivo, queremos el valor de la columna 0.
    # Agresivo = Clase 0
    agg_col_idx = 0 if rf_model.classes_[0] == 0 else 1
    agg_probs = probs[:, agg_col_idx]

    # Suavizado
    my_df['aggression_score'] = pd.Series(agg_probs).rolling(75, center=True).mean().fillna(pd.Series(agg_probs))
    
    # Reglas Híbridas
    threshold = 0.50
    over_limit = my_df['speed_kmh'] > 140
    my_df.loc[over_limit, 'aggression_score'] = 1.0
    
    # Predicción binaria
    my_df['pred_is_aggressive'] = my_df['aggression_score'] > threshold
    
    # Mapeo a Label Original (Asumiendo 0=Agresivo, 1=Eco en tus datos originales)
    # Si predice Agresivo (True), debería ser Label 0. Si Eco (False), Label 1.
    my_df['pred_label_code'] = np.where(my_df['pred_is_aggressive'], 0, 1)

    # Métricas
    acc = accuracy_score(my_df['label'], my_df['pred_label_code'])
    print(f"Accuracy Global: {acc*100:.2f}%")
    print("\n   Reporte de Clasificación:")
    print(classification_report(my_df['label'], my_df['pred_label_code'], target_names=['Agresivo (0)', 'Eco (1)']))

    # 7. VISUALIZACIÓN
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(my_df['timestamp_sec'], my_df['speed_kmh'], color='#38BDF8', label='Velocidad', alpha=0.6)
    aggr_points = my_df[my_df['pred_is_aggressive']]
    plt.scatter(aggr_points['timestamp_sec'], aggr_points['speed_kmh'], color='#EF4444', s=10, label='IA Detecta Agresivo', zorder=5)
    plt.title(f"Validación (Acc: {acc*100:.1f}%)")
    plt.ylabel("km/h")
    plt.legend()
    plt.grid(True, alpha=0.2)

    plt.subplot(2, 1, 2)
    plt.plot(my_df['timestamp_sec'], my_df['aggression_score'], color='purple', label='Score Agresividad')
    plt.axhline(threshold, color='red', linestyle='--', label=f'Umbral {threshold}')
    plt.ylabel("Score (0-1)")
    plt.xlabel("Tiempo (s)")
    plt.legend()
    plt.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('validacion_modelo_v5.png')
    print("   ✅ Gráfica generada: 'validacion_modelo_v5.png'")
    # plt.show() # Comentado para evitar bloqueos si no hay interfaz gráfica

if __name__ == "__main__":
    main()