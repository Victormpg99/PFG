import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from data_loader import CarDataStandardizer
except ImportError:
    print("Error inicial")
    exit()

def main():
    print(" === DIAGNÓSTICO FINO DEL SISTEMA DE INGESTA v6 === ")
    
    loader = CarDataStandardizer()
    
    import glob
    search_pattern = os.path.join("**", "2025-12-29 21-00-43*.csv") 
    files = glob.glob(search_pattern, recursive=True)
    
    if not files:
        files = glob.glob(os.path.join("**", "*.csv"), recursive=True)
        if not files: return

    target_file = max(files, key=os.path.getctime)
    print(f"Archivo: {target_file}")
    
    # Cargamos (El propio loader imprimirá DEBUG de columnas)
    df = loader.load_single_file(target_file, driver_id=999)
    
    if df is None or df.empty:
        print("Data Loader devolvió vacío.")
        return

    print(f"Filas útiles: {len(df)}")
    print("-" * 50)

    # ENERGÍA
    print("[ENERGÍA]")
    if 'fuel_source' in df.columns:
        source = df['fuel_source'].iloc[0]
        print(f"   ► ESTRATEGIA: {source}")
        
        # Validamos datos
        lh = df['fuel_rate_lh']
        print(f"   ► Max L/h: {lh.max():.2f}")
        print(f"   ► Mean L/h: {lh.mean():.2f}")
        print(f"   ► Ceros: {(lh == 0).sum()} / {len(df)}")
    else:
        print(" Error crítico en módulo energía.")

    print("-" * 50)
    
    # SALUD (Debug STFT)
    print("[SALUD]")
    health = loader.analyze_vehicle_health(df)
    print(health)
    
    # Check específico de STFT
    if 'stft' in df.columns:
        print(f"   ► Datos Raw STFT: {df['stft'].dropna().head(5).tolist()}")

if __name__ == "__main__":
    main()