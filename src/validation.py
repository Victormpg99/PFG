import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import unicodedata
from physics_engine import PhysicsEngine

def normalize_str(s):
    """Normaliza nombres de columnas"""
    return ''.join(c for c in unicodedata.normalize('NFD', str(s).lower()) if unicodedata.category(c) != 'Mn')

def find_col(df, keywords):
    for col in df.columns:
        if all(k in normalize_str(col) for k in keywords):
            return col
    return None

def validate_model(file_path):
    print(f"--- ANALIZANDO: {file_path} ---")
    
    # 1. Cargar datos (soporta CSV y Excel/ODS si tienes librerías)
    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1', sep=None, engine='python')
    except:
        df = pd.read_csv(file_path, encoding='utf-8', sep=None, engine='python')

    # 2. Encontrar columnas clave
    col_rpm = find_col(df, ['revoluciones', 'motor'])
    col_map = find_col(df, ['presion', 'absoluta', 'colector']) # O busca 'kpa'
    col_iat = find_col(df, ['temperatura', 'aire', 'colector'])
    col_fuel_cum = find_col(df, ['combustible', 'usado', '(l)']) # Columna L
    col_time = find_col(df, ['time'])

    if not all([col_rpm, col_map, col_iat, col_fuel_cum]):
        print("Error: No se encuentran todas las columnas necesarias.")
        return

    # 3. Preparar datos
    df_val = df.copy()
    
    # Convertir tiempo a segundos relativos
    start_t = pd.to_datetime(df_val[col_time].iloc[0], errors='coerce')
    df_val['seconds'] = (pd.to_datetime(df_val[col_time], errors='coerce') - start_t).dt.total_seconds()
    
    # Asegurar numéricos y ordenar
    cols_sensors = [col_rpm, col_map, col_iat, col_fuel_cum]
    for c in cols_sensors:
        df_val[c] = pd.to_numeric(df_val[c], errors='coerce')
    
    df_val = df_val.sort_values('seconds')
    
    # Rellenar huecos (forward fill) para tener datos continuos del motor
    df_val[[col_rpm, col_map, col_iat]] = df_val[[col_rpm, col_map, col_iat]].ffill()

    # 4. Calcular Consumo Estimado (Física)
    # AJUSTE: Bajamos eficiencia volumétrica a 0.66 basado en el análisis previo
    physics = PhysicsEngine()
    physics.VOL_EFFICIENCY = 0.66 
    
    df_val['est_lh'] = df_val.apply(
        lambda x: physics.estimate_fuel_consumption(x[col_rpm], x[col_map], x[col_iat]), 
        axis=1
    )

    # 5. Integrar para obtener Litros Acumulados (Estimados)
    # Litros = (L/h) * (horas que han pasado)
    df_val['dt_h'] = df_val['seconds'].diff() / 3600.0
    df_val['est_liters_cum'] = (df_val['est_lh'] * df_val['dt_h']).cumsum()

    # 6. Comparar con Litros Reales (Columna L)
    # Alineamos el inicio para que ambas empiecen en 0
    valid_real = df_val.dropna(subset=[col_fuel_cum])
    
    if len(valid_real) > 0:
        # Offset Real
        start_real = valid_real[col_fuel_cum].iloc[0]
        df_val['real_liters_aligned'] = df_val[col_fuel_cum] - start_real
        
        # Offset Estimado (en el mismo instante que empieza el real)
        start_idx = valid_real.index[0]
        offset_est = df_val.loc[start_idx, 'est_liters_cum']
        df_val['est_liters_aligned'] = df_val['est_liters_cum'] - offset_est

        # 7. Generar Gráfica
        plt.figure(figsize=(10, 6))
        
        # Línea continua naranja: Nuestra Estimación
        plt.plot(df_val['seconds'], df_val['est_liters_aligned'], 
                 label='Modelo Físico (Acumulado)', color='orange', linewidth=2)
        
        # Puntos azules: Datos Reales del OBD (Combustible Usado)
        plt.scatter(valid_real['seconds'], df_val.loc[valid_real.index, 'real_liters_aligned'], 
                    label='Dato Real OBD (Combustible Usado)', color='blue', s=15, alpha=0.6)

        plt.title('Validación Final: Consumo Real vs Modelo Físico')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Litros Consumidos')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('validacion_consumo.png')
        print("Gráfica guardada como 'validacion_consumo.png'")
        
        # Imprimir métricas finales
        total_real = df_val.loc[valid_real.index[-1], 'real_liters_aligned']
        total_est = df_val.loc[valid_real.index[-1], 'est_liters_aligned']
        
        print(f"\nRESULTADOS FINALES:")
        print(f"   Litros Reales:   {total_real:.4f} L")
        print(f"   Litros Modelo:   {total_est:.4f} L")
        print(f"   Precisión:       {min(total_real, total_est)/max(total_real, total_est)*100:.1f}%")
        
    else:
        print("No se encontraron datos válidos en la columna de Combustible Usado.")

if __name__ == "__main__":
    # Ajusta la ruta a tu archivo
    validate_model(r'D:\Universidad\00_PFG\TEST\2022-08-02 19-20-44.csv')