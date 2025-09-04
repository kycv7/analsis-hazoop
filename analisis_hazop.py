import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
class HAZOPAnalizer:
    def __init__(self):
        self.deviation=[]
        self.causes=[]
        self.consequences=[]
        self.safeguards=[]
        self.recommendations=[]
    def define_hazop_parameters(self):
        """Definir parámetros y guías HAZOP para el proceso SLES"""
        parameters = {
            'flujo_EO': {'min': 150, 'max': 250, 'units': 'kg/h'},
            'flujo_acid': {'min': 200, 'max': 300, 'units': 'kg/h'},
            'temp_reactor': {'min': 35, 'max': 45, 'units': '°C'},
            'presion_reactor': {'min': 1.5, 'max': 2.5, 'units': 'bar'},
            'ph_neutralizacion': {'min': 6.5, 'max': 7.5, 'units': 'pH'},
            'conc_sLES': {'min': 68, 'max': 72, 'units': '%'}
        }
    def generate_hazop_table(self):
        """Generar tabla HAZOP completa"""
        hazop_data = [
            # Sulfatación
            {'node': 'Reactor Sulfatación', 'parameter': 'Flujo EO', 'guide_word': 'MÁS', 
             'deviation': 'Exceso de Óxido de Etileno', 'cause': 'Válvula FO-101 atascada abierta',
             'consequence': 'Sobre-sulfatación, exotérmica descontrolada', 'risk_level': 'Alto',
             'safeguard': 'SIS-201, alarmas de alto flujo', 'action': 'Revisar válvula FO-101 mensualmente'},
            
            {'node': 'Reactor Sulfatación', 'parameter': 'Temperatura', 'guide_word': 'MÁS', 
             'deviation': 'Temperatura elevada', 'cause': 'Fallo en sistema de refrigeración',
             'consequence': 'Degradación del producto, riesgo de runaway', 'risk_level': 'Alto',
             'safeguard': 'TISA-301, parada de emergencia', 'action': 'Mantenimiento preventivo chiller'},
            
            # Neutralización
            {'node': 'Neutralizador', 'parameter': 'pH', 'guide_word': 'MENOS', 
             'deviation': 'pH bajo (ácido)', 'cause': 'Fallo en dosificación de NaOH',
             'consequence': 'Corrosión, producto fuera de especificación', 'risk_level': 'Medio',
             'safeguard': 'Controlador pH-401 con alarmas', 'action': 'Calibración semanal de pH-metro'},
            
            {'node': 'Tanque Mezcla', 'parameter': 'Concentración', 'guide_word': 'NO', 
             'deviation': 'No hay concentración de SLES', 'cause': 'Parada de agitador MX-201',
             'consequence': 'Producto heterogéneo, fuera de especificación', 'risk_level': 'Medio',
             'safeguard': 'Interlock agitador-concentración', 'action': 'Monitorizar estado del agitador'}
        ]
        return pd.DataFrame(hazop_data)
# Ejecutar análisis HAZOP
hazop = HAZOPAnalizer()
hazop_table = hazop.generate_hazop_table()
print("Tabla HAZOP generada:")
print(hazop_table.head())

class ProcessDataSimulator:
    def __init__(self, days=30, samples_per_hour=12):
        self.days = days
        self.samples = samples_per_hour
        self.time_index = None
        
    def generate_time_index(self):
        """Generar índice de tiempo para la simulación"""
        start_date = datetime(2024, 1, 1)
        end_date = start_date + timedelta(days=self.days)
        return pd.date_range(start=start_date, end=end_date, 
                           freq=f'{60//self.samples}min')[:-1]
    
    def simulate_normal_operation(self):
        """Simular operación normal del proceso"""
        time_index = self.generate_time_index()
        n_samples = len(time_index)
        
        data = {
            'timestamp': time_index,
            'flujo_EO': np.random.normal(200, 10, n_samples),
            'flujo_acid': np.random.normal(250, 15, n_samples),
            'temp_reactor': np.random.normal(40, 1.5, n_samples),
            'presion_reactor': np.random.normal(2.0, 0.2, n_samples),
            'ph_neutralizacion': np.random.normal(7.0, 0.3, n_samples),
            'conc_sLES': np.random.normal(70, 1.0, n_samples),
            'agitador_rpm': np.random.normal(120, 5, n_samples),
            'nivel_reactor': np.random.normal(75, 5, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def inject_anomalies(self, df_normal):
        """Inyectar anomalías basadas en escenarios HAZOP"""
        df = df_normal.copy()
        n_samples = len(df)
        
        # Anomalía 1: Exceso de EO (válvula atascada)
        anomaly_start_1 = n_samples // 4
        anomaly_end_1 = anomaly_start_1 + 24  # 2 horas de anomalía
        df.loc[anomaly_start_1:anomaly_end_1, 'flujo_EO'] = np.random.normal(280, 5, anomaly_end_1 - anomaly_start_1 + 1)
        df.loc[anomaly_start_1:anomaly_end_1, 'temp_reactor'] += np.random.normal(3, 0.5, anomaly_end_1 - anomaly_start_1 + 1)
        
        # Anomalía 2: pH bajo (fallo en neutralización)
        anomaly_start_2 = n_samples // 2
        anomaly_end_2 = anomaly_start_2 + 36  # 3 horas de anomalía
        df.loc[anomaly_start_2:anomaly_end_2, 'ph_neutralizacion'] = np.random.normal(5.8, 0.2, anomaly_end_2 - anomaly_start_2 + 1)
        df.loc[anomaly_start_2:anomaly_end_2, 'conc_sLES'] -= np.random.normal(3, 0.5, anomaly_end_2 - anomaly_start_2 + 1)
        
        # Anomalía 3: Agitador fallando
        anomaly_start_3 = 3 * n_samples // 4
        anomaly_end_3 = anomaly_start_3 + 18  # 1.5 horas de anomalía
        df.loc[anomaly_start_3:anomaly_end_3, 'agitador_rpm'] = np.random.normal(60, 10, anomaly_end_3 - anomaly_start_3 + 1)
        df.loc[anomaly_start_3:anomaly_end_3, 'conc_sLES'] += np.random.uniform(-4, 4, anomaly_end_3 - anomaly_start_3 + 1)
        
        # Añadir etiquetas de anomalía
        df['anomaly'] = 0
        df.loc[anomaly_start_1:anomaly_end_1, 'anomaly'] = 1
        df.loc[anomaly_start_2:anomaly_end_2, 'anomaly'] = 2
        df.loc[anomaly_start_3:anomaly_end_3, 'anomaly'] = 3
        
        return df

# Generar datos de simulación
simulator = ProcessDataSimulator(days=60, samples_per_hour=6)
normal_data = simulator.simulate_normal_operation()
process_data = simulator.inject_anomalies(normal_data)

print(f"Datos del proceso generados: {len(process_data)} muestras")
print(process_data.head())


