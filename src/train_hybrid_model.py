import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

try:
    from data_loader import CarDataStandardizer
except ImportError:
    print("Error inicial")
    exit()

def train_compare_models():
    print("INICIANDO COMPARACION DE MODELOS FINAL")
    
    loader = CarDataStandardizer()
    full_df = loader.load_training_dataset(os.getcwd())
    
    if full_df is None: return

    # Variables para cada modelo
    feats_A = ['speed_kmh', 'rpm']
    feats_B = ['accel_ms2', 'jerk_ms3', 'accel_std_3s', 'speed_std_3s']
    feats_C = feats_A + feats_B + ['power_index', 'risk_index'] # El "Ingeniero"
    
    target = 'label'
    y = full_df[target]
    
    # Preparamos Split fijo para todos
    # Incluimos todas las columnas en X para poder filtrar después
    X_train, X_test, y_train, y_test = train_test_split(full_df, y, test_size=0.2, random_state=42, stratify=y)

    results = []

    # --- RONDA 1: MODELOS PUROS (ML) ---
    models_config = [
        ("Modelo A (Solo Velocidad)", feats_A),
        ("Modelo C (IA)", feats_C)
    ]

    trained_models = {} # Guardamos los modelos entrenados

    for name, feats in models_config:
        print(f"\n Entrenando {name}...")
        clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, class_weight='balanced')
        # Entrenamos solo con las columnas correspondientes
        clf.fit(X_train[feats], y_train)
        
        y_pred = clf.predict(X_test[feats])
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label=0) # 0=Agresivo
        
        results.append({'Modelo': name, 'Accuracy': acc, 'F1_Agresivo': f1})
        trained_models[name] = clf # Guardamos para usarlo en el D

    # --- RONDA 2: MODELO D---
    print(f"\n Entrenando Modelo D (Regla > 140km/h + Modelo C)...")
    
    # 1. Cogemos las predicciones del Modelo C 
    model_c = trained_models["Modelo C"]
    base_preds = model_c.predict(X_test[feats_C])
    
    # 2. APLICAMOS REGLA ADICIONAL
    # Si velocidad > 140, forzamos la predicción a 0 (Agresivo)
    # Convertimos a Series para trabajar cómodamente
    final_preds = pd.Series(base_preds, index=X_test.index)
    
    # Buscamos índices donde se viola el límite
    speeding_indices = X_test[X_test['speed_kmh'] > 140].index
    
    # Sobrescribimos
    final_preds.loc[speeding_indices] = 0 
    
    # Evaluamos
    acc_d = accuracy_score(y_test, final_preds)
    f1_d = f1_score(y_test, final_preds, pos_label=0)
    
    print(f"REGLA APLICADA: {len(speeding_indices)} muestras corregidas por exceso de velocidad.")
    
    results.append({'Modelo': 'Modelo D', 'Accuracy': acc_d, 'F1_Agresivo': f1_d})

    # --- VISUALIZACIÓN ---
    df_res = pd.DataFrame(results)
    print("\nTABLA FINAL DE RESULTADOS:")
    print(df_res)
    
    # Gráfica
    plt.figure(figsize=(10, 6))
    x = np.arange(len(df_res))
    plt.bar(x, df_res['Accuracy'], color=['gray', 'skyblue', 'gold'], width=0.5)
    plt.xticks(x, df_res['Modelo'])
    plt.ylim(0.5, 1.0) # Zoom para ver diferencias
    plt.ylabel('Precisión Global')
    plt.title('Impacto de la Regla (>140 km/h)')
    
    for i, v in enumerate(df_res['Accuracy']):
        plt.text(i, v + 0.01, f"{v*100:.1f}%", ha='center', fontweight='bold')
        
    plt.savefig('comparativa_final_expert.png')
    print("Gráfica guardada.")

if __name__ == "__main__":
    train_compare_models()