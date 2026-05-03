import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'src'))
from data_loader import CarDataStandardizer

def main():
    print(" === ANÁLISIS DE DATOS CRUDOS === ")
    
    # 1. Buscamos el archivo
    import glob
    files = glob.glob(os.path.join("**", "2025-12-29 21-00-43*.csv"), recursive=True)
    if not files: return
    target = files[0]
    print(f" Objetivo: {target}")

    # 2. Carga en bruto (sin procesar)
    try:
        df_raw = pd.read_csv(target, encoding='ISO-8859-1', low_memory=False)
    except:
        df_raw = pd.read_csv(target, encoding='utf-8', low_memory=False)

    print(f" Filas Totales: {len(df_raw)}")
    print("-" * 40)

    # 3. INSPECCIÓN DE COLUMNAS CLAVE
    # Definimos las columnas que el sistema detectó antes
    cols_to_check = {
        'RPM': [c for c in df_raw.columns if 'Revoluciones' in c and 'rpm' in c],
        'MAP': [c for c in df_raw.columns if 'Presi' in c and 'colector' in c],
        'L/100': [c for c in df_raw.columns if 'Consumo' in c and '100' in c and 'instant' in c]
    }

    loader = CarDataStandardizer()

    for name, candidates in cols_to_check.items():
        print(f"\n INSPECCIONANDO: {name}")
        if not candidates:
            print(" Columna no encontrada en el CSV original.")
            continue
            
        col_name = candidates[0] # Cogemos la primera coincidencia
        print(f" Columna: '{col_name}'")
        
        # Muestreo de datos
        series = df_raw[col_name]
        non_nulls = series.notna().sum()
        print(f"   No Nulos: {non_nulls} / {len(df_raw)}")
        
        # Muestra 5 valores que NO sean nulos (si existen)
        sample = series.dropna().head(5).tolist()
        print(f"   Muestra Raw (String): {sample}")
        
        # Prueba de conversión numérica
        numeric_sample = loader._safe_numeric(series).dropna()
        valid_numeric = (numeric_sample > 0).sum()
        print(f"   Valores Numéricos > 0: {valid_numeric}")
        
        if valid_numeric == 0:
            print(" DIAGNÓSTICO: La columna existe pero está vacía o son ceros.")
        else:
            print(f" DIAGNÓSTICO: Datos válidos detectados. (Ej: {numeric_sample.iloc[0] if not numeric_sample.empty else '?'})")

if __name__ == "__main__":
    main()