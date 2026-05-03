import pandas as pd
import numpy as np
import glob
import os
import unicodedata

try:
    from physics_engine import PhysicsEngine
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from physics_engine import PhysicsEngine

class CarDataStandardizer:
    def __init__(self):
        self.physics = PhysicsEngine()

    def _normalize_str(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', str(s).lower()) if unicodedata.category(c) != 'Mn')

    def _find_column(self, df, keywords, exclude=None, prefer_densest=True):
        candidates = []
        for col in df.columns:
            col_norm = self._normalize_str(col)
            if all(self._normalize_str(k) in col_norm for k in keywords):
                if exclude and any(self._normalize_str(bad) in col_norm for bad in exclude):
                    continue
                candidates.append(col)
        
        if not candidates: return None
        
        if prefer_densest:
            return max(candidates, key=lambda c: df[c].notna().sum())
        else:
            candidates.sort(key=lambda x: len(x))
            return candidates[0]

    def _safe_numeric(self, series):
        if series is None: return pd.Series(0)
        return pd.to_numeric(series.astype(str).str.replace(',', '.'), errors='coerce')

    def load_single_file(self, path, driver_id=0):
        try:
            df = pd.read_csv(path, encoding='ISO-8859-1', low_memory=False)
        except:
            try:
                df = pd.read_csv(path, encoding='utf-8', low_memory=False)
            except:
                df = pd.read_csv(path, encoding='cp1252', low_memory=False)

        # 1. IDENTIFICACIÓN DE COLUMNAS AMPLIANDO VARIABLES
        col_speed = self._find_column(df, ['velocidad', 'km/h'], prefer_densest=True)
        col_rpm = self._find_column(df, ['rpm'], exclude=['power', 'torque'], prefer_densest=True) or \
                  self._find_column(df, ['revoluciones'], prefer_densest=True)
        col_time = self._find_column(df, ['time']) 
        col_label = 'Label'
        
        col_coolant = self._find_column(df, ['temperatura', 'liquido']) or \
                      self._find_column(df, ['temperatura', 'enfriamiento']) or \
                      self._find_column(df, ['refrigerante']) or \
                      self._find_column(df, ['coolant'])
        
        col_load = self._find_column(df, ['carga', 'motor']) or \
                   self._find_column(df, ['valor', 'carga']) or \
                   self._find_column(df, ['engine', 'load'])

        # --- CORRECCIÓN VOLTAJE: PRIORIDAD Y EXCLUSIÓN ---
        # 1. Busca explícitamente ELM o Módulo de Control
        # 2. Si busca genérico 'voltaje', EXCLUYE 'sensor' y 'oxigeno'
        col_volt = self._find_column(df, ['voltaje', 'elm']) or \
                   self._find_column(df, ['voltaje', 'modulo']) or \
                   self._find_column(df, ['voltaje', 'control']) or \
                   self._find_column(df, ['voltaje'], exclude=['oxigeno', 'oxygen', 'sensor', 'lambda', 'banco']) or \
                   self._find_column(df, ['voltage'], exclude=['oxygen', 'sensor'])

        col_stft = self._find_column(df, ['ajuste', 'corto']) or self._find_column(df, ['short', 'trim'])
        
        col_fuel_rate = self._find_column(df, ['consumo', 'l/h']) or self._find_column(df, ['fuel', 'rate'])
        col_fuel_l100 = self._find_column(df, ['consumo', 'l/100'], exclude=['medio', 'avg'])
        col_lat, col_lon = self._find_column(df, ['lat']), self._find_column(df, ['lon'])

        if not col_speed: return None

        # 2. VALIDACIÓN DE DENSIDAD
        DENSITY_THRESHOLD = 0.0005 
        total_rows = len(df)
        
        if col_rpm:
            valid_ratio = self._safe_numeric(df[col_rpm]).notna().sum() / total_rows
            if valid_ratio < DENSITY_THRESHOLD: col_rpm = None

        # 3. SINCRONIZACIÓN
        health_sensors = [c for c in [col_load, col_volt, col_stft, col_coolant] if c]
        df_filled = df.copy()
        
        if health_sensors:
            df_filled[health_sensors] = df_filled[health_sensors].ffill().bfill()
            
        dyn_sensors = [c for c in [col_speed, col_rpm, col_fuel_rate] if c]
        if dyn_sensors:
            df_filled[dyn_sensors] = df_filled[dyn_sensors].ffill(limit=30).bfill(limit=30)

        # 4. CONSTRUCCIÓN
        new_df = pd.DataFrame()
        new_df['speed_kmh'] = self._safe_numeric(df_filled[col_speed])
        new_df['rpm'] = self._safe_numeric(df_filled[col_rpm]).fillna(0) if col_rpm else 0
        if col_label in df.columns: new_df['label'] = pd.to_numeric(df[col_label], errors='coerce')

        if col_time:
            try:
                times = pd.to_datetime(df[col_time], errors='coerce', format='mixed')
                new_df['timestamp_sec'] = (times - times.dropna().iloc[0]).dt.total_seconds()
            except: new_df['timestamp_sec'] = np.arange(len(new_df)) * 0.068
        else:
            new_df['timestamp_sec'] = np.arange(len(new_df)) * 0.068

        new_df['accel_ms2'] = self.physics.calculate_acceleration(new_df)
        new_df['jerk_ms3'] = new_df['accel_ms2'].diff().fillna(0)

        fuel_lh = pd.Series(np.nan, index=new_df.index)
        source = "None"
        if col_fuel_rate:
            fuel_lh = self._safe_numeric(df_filled[col_fuel_rate])
            source = "OBD Direct"
        elif col_fuel_l100:
            l100 = self._safe_numeric(df_filled[col_fuel_l100])
            fuel_lh = l100 * new_df['speed_kmh'] / 100
            source = "Derived L/100"
            
        kinematic = new_df.apply(lambda x: self.physics.estimate_consumption_kinematic(x['speed_kmh'], x['accel_ms2']), axis=1)
        fuel_lh = fuel_lh.fillna(kinematic)
        fuel_lh = np.where((fuel_lh == 0) & (new_df['speed_kmh'] > 10), kinematic, fuel_lh)
        
        new_df['fuel_rate_lh'] = fuel_lh
        new_df['fuel_source'] = source
        new_df['fuel_l100km'] = np.where(new_df['speed_kmh']>5, (new_df['fuel_rate_lh']/new_df['speed_kmh'])*100, 0)

        mapping = {'engine_load': col_load, 'voltage': col_volt, 'stft': col_stft, 'coolant_temp': col_coolant, 'latitude': col_lat, 'longitude': col_lon}
        for k, v in mapping.items():
            new_df[k] = self._safe_numeric(df_filled[v]) if v else np.nan

        window = 44
        new_df['accel_std_3s'] = new_df['accel_ms2'].rolling(window).std().fillna(0)
        new_df['speed_std_3s'] = new_df['speed_kmh'].rolling(window).std().fillna(0)
        new_df['power_index'] = new_df['speed_kmh'] * new_df['accel_ms2'].abs()
        new_df['risk_index'] = new_df['speed_kmh'] * new_df['jerk_ms3'].abs()
        
        return new_df.dropna(subset=['speed_kmh', 'timestamp_sec'])

    def load_training_dataset(self, folder_path):
        files = glob.glob(os.path.join(folder_path, '**', '*Classified*.csv'), recursive=True)
        if not files: return None
        all_data = []
        for f in files:
            d_id = 1 if 'driver1' in f else 2
            df = self.load_single_file(f, driver_id=d_id)
            if df is not None and 'label' in df.columns: all_data.append(df)
        return pd.concat(all_data, ignore_index=True) if all_data else None
    
    
# --- TEST RÁPIDO ---
if __name__ == "__main__":
    print("Test de Data Loader...")
    loader = CarDataStandardizer()
    
    my_car = r"D:\Universidad\00_PFG\PFG_BusCan\TEST\nuevos_datos\2025-12-29 17-28-50.csv"
    if os.path.exists(my_car):
        print(f"Probando los datos cargados: {os.path.basename(my_car)}")
        df = loader.load_single_file(my_car)
        print(f" Filas cargadas: {len(df)}")
        print(f" Columnas nuevas: {[c for c in df.columns if '3s' in c]}")
    else:
        print("No se ha encontrado el archivo")