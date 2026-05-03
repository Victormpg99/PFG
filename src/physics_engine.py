import numpy as np
import pandas as pd

class PhysicsEngine:
    def __init__(self):
        # Parámetros térmicos y mecánicos (Motor 1.4 TSI o similar)
        self.DISPLACEMENT = 0.0014   # m^3 (1.4L)
        self.R_AIR = 287.05          # Constante gases aire
        self.STOI_RATIO = 14.7       # Relación estequiométrica gasolina
        self.GAS_DENSITY = 740       # kg/m^3
        self.VOL_EFFICIENCY = 0.66   # Eficiencia volumétrica estimada
        
        # Parámetros Cinéticos (Para 'Rescate' de datos perdidos)
        self.VEHICLE_MASS = 1450     # kg (Coche + Ocupantes)
        self.CD = 0.32               # Coeficiente aerodinámico
        self.FRONTAL_AREA = 2.2      # m^2
        self.ROLLING_RESISTANCE = 0.015 
        self.AIR_DENSITY_EXT = 1.225 # kg/m^3
        
        # Factor de conversión Energía -> Litros (Gasolina ~8.9 kWh/L, Eficiencia Motor ~25%)
        self.KWH_PER_LITER = 8.9
        self.ENGINE_EFFICIENCY = 0.25 

        # Límites de seguridad física
        self.MAX_ACCEL = 6.0         # m/s^2

    def estimate_fuel_consumption(self, rpm, map_kpa, temp_c=25):
        """Método Speed-Density: Preciso si hay sensores MAP y RPM."""
        if pd.isna(rpm) or pd.isna(map_kpa) or rpm <= 300: 
            return 0.6 # Consumo mínimo ralentí
            
        if pd.isna(temp_c): temp_c = 25
        temp_k = temp_c + 273.15
        
        # Densidad aire en colector
        air_density = (map_kpa * 1000) / (self.R_AIR * temp_k)
        # Flujo másico (kg/s)
        maf_kgs = (self.DISPLACEMENT * self.VOL_EFFICIENCY * rpm * air_density) / 120
        # Flujo combustible (L/h)
        fuel_lh = ((maf_kgs * 1000) / self.STOI_RATIO * 3600) / self.GAS_DENSITY
        return max(0.6, fuel_lh)

    def estimate_consumption_kinematic(self, speed_kmh, accel_ms2):
        """
        Método de Rescate: Estima consumo basándose en la fuerza necesaria para mover el coche.
        Se usa cuando fallan los sensores OBD.
        """
        if speed_kmh < 1: return 0.6 # Ralentí
        
        v_ms = speed_kmh / 3.6
        
        # Fuerzas opuestas al movimiento
        f_aero = 0.5 * self.AIR_DENSITY_EXT * self.CD * self.FRONTAL_AREA * (v_ms**2)
        f_roll = self.VEHICLE_MASS * 9.81 * self.ROLLING_RESISTANCE
        f_inertial = self.VEHICLE_MASS * max(0, accel_ms2) # Solo gastamos al acelerar
        
        total_force = f_aero + f_roll + f_inertial
        
        # Potencia necesaria (kW)
        power_kw = (total_force * v_ms) / 1000.0
        
        # Litros estimados
        fuel_lh = power_kw / (self.KWH_PER_LITER * self.ENGINE_EFFICIENCY)
        
        return max(0.6, fuel_lh)

    def calculate_acceleration(self, df):
        """Derivada de velocidad con suavizado"""
        if 'speed_kmh' not in df.columns or 'timestamp_sec' not in df.columns:
            return pd.Series(0, index=df.index)

        dt_series = df['timestamp_sec'].diff()
        median_dt = dt_series.median()
        if pd.isna(median_dt) or median_dt == 0: median_dt = 0.068

        # Ventana de 3 segundos para eliminar ruido GPS
        window_size = max(1, int(3.0 / median_dt))
        speed_smooth = df['speed_kmh'].rolling(window=window_size, center=True, min_periods=1).mean()
        
        v_ms = speed_smooth / 3.6
        accel = v_ms.diff() / dt_series

        return accel.fillna(0).clip(-self.MAX_ACCEL, self.MAX_ACCEL)

