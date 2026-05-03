import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DriverProfiler:
    def __init__(self):
        # --- AJUSTES DE CALIBRACIÓN ---
        # Basado en la validación cruzada con el Dataset Driver 2
        # Umbral subido a 3.5 para coincidir con el criterio del dataset
        self.HARD_ACCEL_THRESHOLD = 3.5  
        self.HARD_BRAKE_THRESHOLD = -3.5
        self.HIGH_RPM_THRESHOLD = 3500   
        self.IDLING_RPM = 950

    def classify_moment(self, row):
        """
        Recibe una fila con datos físicos y devuelve una etiqueta de estilo.
        Retorna: 'Aggressive_Accel', 'Hard_Braking', 'Inefficient_HighRPM', 'Idling', 'Normal_Eco'
        """
        accel = row['accel_ms2']
        rpm = row['rpm']
        speed = row['speed_kmh']

        # 1. Agresividad (Fuerzas G)
        if accel > self.HARD_ACCEL_THRESHOLD: return 'Aggressive_Accel'
        elif accel < self.HARD_BRAKE_THRESHOLD: return 'Hard_Braking'
        
        # 2. Ineficiencia
        if rpm > self.HIGH_RPM_THRESHOLD and abs(accel) < 1.0: return 'Inefficient_HighRPM'
        
        # 3. Ralentí
        if speed < 2 and rpm > 0: return 'Idling'
        
        # 4. Normal
        return 'Normal_Eco'

    def analyze_trip(self, df):
        """Método completo para generar el informe de un viaje."""
        # Aseguramos que tenemos las etiquetas
        df['driving_style'] = df.apply(self.classify_moment, axis=1)
        
        counts = df['driving_style'].value_counts()
        total_samples = len(df)
        
        # Sistema de Puntuación (Score)
        score = 100.0
        penalties = {
            'Aggressive_Accel': 1.0,    
            'Hard_Braking': 1.0,
            'Inefficient_HighRPM': 0.5,
            'Idling': 0.1
        }

        print("\n--- DESGLOSE DEL PERFIL ---")
        for style, count in counts.items():
            pct = (count / total_samples) * 100
            print(f" > {style}: {pct:.2f}% del tiempo")
            if style in penalties:
                score -= pct * penalties[style]

        return max(0, min(100, score)), df

if __name__ == "__main__":
    from data_loader import CarDataStandardizer
    from physics_engine import PhysicsEngine

    loader = CarDataStandardizer()
    physics = PhysicsEngine()
    profiler = DriverProfiler()

    path = r'D:\Universidad\00_PFG\TEST\Data_Driver2.csv'
    
    try:
        print("Cargando datos...")
        df = loader.load_seat_leon_dataset(path)
        
        print("Aplicando Motor Físico (Consumo + Aceleración Suavizada)...")
        # 1. Consumo
        df['fuel_rate_lh'] = df.apply(
            lambda x: physics.estimate_fuel_consumption(
                x['rpm'], 
                x['intake_pressure_kpa'], 
                x.get('intake_temp_c', 25)
            ), axis=1
        )
        
        # 2. Aceleración (Automática con suavizado 3s)
        df['accel_ms2'] = physics.calculate_acceleration(df)

        print("Juzgando al conductor...")
        final_score, df_labeled = profiler.analyze_trip(df)

        print(f"\nPUNTUACIÓN FINAL: {final_score:.1f} / 100")
        
        # Generar Gráfico
        plt.figure(figsize=(10, 6))
        df_labeled['driving_style'].value_counts().plot.pie(autopct='%1.1f%%', cmap='Set3')
        plt.title(f'Perfil de Conducción (Score: {final_score:.1f})')
        plt.ylabel('')
        plt.savefig('driving_profile_pie.png')
        print("Informe guardado en 'driving_profile_pie.png'")

    except Exception as e:
        print(f"Error: {e}")