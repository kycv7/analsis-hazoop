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
