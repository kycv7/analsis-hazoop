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

from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.features = ['flujo_EO', 'flujo_acid', 'temp_reactor', 
                        'presion_reactor', 'ph_neutralizacion', 'conc_sLES',
                        'agitador_rpm', 'nivel_reactor']
    
    def preprocess_data(self, df):
        """Preprocesar datos para el modelo"""
        X = df[self.features].copy()
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled
    
    def train_isolation_forest(self, X):
        """Entrenar modelo Isolation Forest"""
        iso_forest = IsolationForest(
            n_estimators=100,
            contamination=0.05,  # 5% de anomalías esperadas
            random_state=42
        )
        iso_forest.fit(X)
        return iso_forest
    
    def detect_anomalies(self, df):
        """Detectar anomalías en los datos del proceso"""
        X = self.preprocess_data(df)
        
        # Entrenar y predecir con Isolation Forest
        iso_model = self.train_isolation_forest(X)
        predictions = iso_model.predict(X)
        
        # Añadir resultados al DataFrame
        df['anomaly_score'] = iso_model.decision_function(X)
        df['is_anomaly'] = predictions == -1
        
        return df
    
    def calculate_process_stability_index(self, df, window_size=12):
        """Calcular índice de estabilidad del proceso"""
        df = df.copy()
        
        # Calcular variabilidad en ventana móvil
        for feature in self.features:
            rolling_std = df[feature].rolling(window=window_size).std()
            df[f'{feature}_stability'] = 1 / (1 + rolling_std)
        
        # Índice de estabilidad general
        stability_cols = [f'{feat}_stability' for feat in self.features]
        df['process_stability'] = df[stability_cols].mean(axis=1)
        
        return df
detector = AnomalyDetector()
process_data_with_anomalies = detector.detect_anomalies(process_data)
process_data_with_stability = detector.calculate_process_stability_index(process_data_with_anomalies)

print("Datos con detección de anomalías:")
print(process_data_with_anomalies[['timestamp', 'anomaly_score', 'is_anomaly']].head())

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import shap

class RootCauseAnalyzer:
    def __init__(self):
        self.explainer = None
        self.feature_importance = None
        self.anomaly_thresholds = {
            'flujo_EO': {'high': 260, 'low': 150},
            'temp_reactor': {'high': 43, 'low': 35},
            'ph_neutralizacion': {'high': 7.5, 'low': 6.2},
            'conc_sLES': {'high': 73, 'low': 67},
            'agitador_rpm': {'high': 140, 'low': 100},
            'presion_reactor': {'high': 2.5, 'low': 1.5}
        }
        
    def _classify_anomaly(self, sample):
        """Clasificar el tipo de anomalía basado en las desviaciones"""
        anomaly_types = []
        
        # Verificar cada parámetro contra sus límites
        if sample['flujo_EO'] > self.anomaly_thresholds['flujo_EO']['high']:
            anomaly_types.append("EXCESO_EO")
        elif sample['flujo_EO'] < self.anomaly_thresholds['flujo_EO']['low']:
            anomaly_types.append("DEFICIT_EO")
            
        if sample['temp_reactor'] > self.anomaly_thresholds['temp_reactor']['high']:
            anomaly_types.append("SOBRETEMPERATURA")
        elif sample['temp_reactor'] < self.anomaly_thresholds['temp_reactor']['low']:
            anomaly_types.append("BAJA_TEMPERATURA")
            
        if sample['ph_neutralizacion'] > self.anomaly_thresholds['ph_neutralizacion']['high']:
            anomaly_types.append("PH_ALCALINO")
        elif sample['ph_neutralizacion'] < self.anomaly_thresholds['ph_neutralizacion']['low']:
            anomaly_types.append("PH_ACIDO")
            
        if sample['conc_sLES'] > self.anomaly_thresholds['conc_sLES']['high']:
            anomaly_types.append("CONCENTRACION_ALTA")
        elif sample['conc_sLES'] < self.anomaly_thresholds['conc_sLES']['low']:
            anomaly_types.append("CONCENTRACION_BAJA")
            
        if sample['agitador_rpm'] > self.anomaly_thresholds['agitador_rpm']['high']:
            anomaly_types.append("AGITACION_EXCESIVA")
        elif sample['agitador_rpm'] < self.anomaly_thresholds['agitador_rpm']['low']:
            anomaly_types.append("AGITACION_INSUFICIENTE")
            
        if sample['presion_reactor'] > self.anomaly_thresholds['presion_reactor']['high']:
            anomaly_types.append("SOBREPRESION")
        elif sample['presion_reactor'] < self.anomaly_thresholds['presion_reactor']['low']:
            anomaly_types.append("BAJA_PRESION")
        
        # Clasificación general basada en la combinación de anomalías
        if not anomaly_types:
            return "ANOMALIA_NO_IDENTIFICADA"
        
        # Priorizar anomalías críticas
        if "EXCESO_EO" in anomaly_types and "SOBRETEMPERATURA" in anomaly_types:
            return "RIESGO_RUNAWAY_REACTIVO"
        elif "PH_ACIDO" in anomaly_types and "CONCENTRACION_BAJA" in anomaly_types:
            return "FALLA_NEUTRALIZACION"
        elif "AGITACION_INSUFICIENTE" in anomaly_types:
            return "FALLA_MEZCLADO"
        elif "SOBREPRESION" in anomaly_types:
            return "RIESGO_PRESION"
        else:
            return "ANOMALIA_MULTIPARAMETRO_" + "_".join(anomaly_types[:3])
    
    def _get_critical_parameters(self, sample):
        """Obtener parámetros críticos que están fuera de rango"""
        critical_params = {}
        
        for param, thresholds in self.anomaly_thresholds.items():
            if param in sample:
                value = sample[param]
                if value > thresholds['high']:
                    critical_params[param] = {
                        'value': value, 
                        'limit': thresholds['high'],
                        'deviation': f"+{value - thresholds['high']:.2f}"
                    }
                elif value < thresholds['low']:
                    critical_params[param] = {
                        'value': value, 
                        'limit': thresholds['low'],
                        'deviation': f"{value - thresholds['low']:.2f}"
                    }
        
        return critical_params
    
    def _get_recommended_actions(self, sample):
        """Generar acciones recomendadas basadas en el tipo de anomalía"""
        anomaly_type = self._classify_anomaly(sample)
        actions = []
        
        # Acciones específicas por tipo de anomalía
        if "EXCESO_EO" in anomaly_type:
            actions.extend([
                "Verificar válvula de control FO-101",
                "Revisar setpoint del controlador de flujo",
                "Monitorear temperatura del reactor cada 5 minutos"
            ])
        
        if "SOBRETEMPERATURA" in anomaly_type:
            actions.extend([
                "Activar sistema de enfriamiento de emergencia",
                "Verificar funcionamiento del intercambiador de calor",
                "Preparar protocolo de parada segura"
            ])
        
        if "PH_ACIDO" in anomaly_type:
            actions.extend([
                "Calibrar sensor de pH inmediatamente",
                "Verificar dosificación de NaOH",
                "Ajustar bomba dosificadora de neutralizante"
            ])
        
        if "AGITACION_INSUFICIENTE" in anomaly_type:
            actions.extend([
                "Verificar motor y transmisión del agitador",
                "Inspeccionar estado de las palas del agitador",
                "Monitorear consumo eléctrico del motor"
            ])
        
        if "SOBREPRESION" in anomaly_type:
            actions.extend([
                "Verificar válvulas de alivio de presión",
                "Inspeccionar sistema de ventilación",
                "Preparar descarga segura si es necesario"
            ])
        
        # Acciones generales
        actions.extend([
            "Notificar al supervisor de turno",
            "Documentar evento en sistema de gestión",
            "Programar mantenimiento preventivo relacionado"
        ])
        
        return actions
    
    def train_cause_classifier(self, df):
        """Entrenar clasificador para identificar causas"""
        # Preparar datos para clasificación
        X = df[detector.features].copy()
        y = df['anomaly'].apply(lambda x: 1 if x > 0 else 0)  # Binario: anomalía vs normal
        
        # Entrenar Random Forest
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        
        # Calcular importancia de características
        importance = pd.DataFrame({
            'feature': detector.features,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return clf, importance
    
    def explain_anomalies_shap(self, df, model):
        """Explicar anomalías usando SHAP"""
        X = df[detector.features].copy()
        
        # Crear explainer SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        return explainer, shap_values
    
    def generate_root_cause_analysis(self, df, anomaly_indices):
        """Generar análisis de causa raíz para anomalías específicas"""
        causes = []
        
        for idx in anomaly_indices:
            if idx < len(df):
                sample = df.iloc[idx]
                
                # Identificar posibles causas basadas en HAZOP
                possible_causes = self._match_hazop_causes(sample)
                
                causes.append({
                    'timestamp': sample['timestamp'],
                    'anomaly_type': self._classify_anomaly(sample),
                    'possible_causes': possible_causes,
                    'critical_parameters': self._get_critical_parameters(sample),
                    'recommended_actions': self._get_recommended_actions(sample)
                })
        
        return pd.DataFrame(causes)
    
    def _match_hazop_causes(self, sample):
        """Emparejar con causas HAZOP"""
        causes = []
        
        if sample['flujo_EO'] > self.anomaly_thresholds['flujo_EO']['high']:
            causes.append("Posible válvula EO atascada - Revisar FO-101")
        if sample['temp_reactor'] > self.anomaly_thresholds['temp_reactor']['high']:
            causes.append("Temperatura elevada - Verificar sistema refrigeración")
        if sample['ph_neutralizacion'] < self.anomaly_thresholds['ph_neutralizacion']['low']:
            causes.append("pH bajo - Revisar dosificación NaOH")
        if sample['agitador_rpm'] < self.anomaly_thresholds['agitador_rpm']['low']:
            causes.append("Problema con agitador - Verificar MX-201")
        if sample['presion_reactor'] > self.anomaly_thresholds['presion_reactor']['high']:
            causes.append("Sobre presión - Verificar válvulas de alivio")
        if sample['conc_sLES'] < self.anomaly_thresholds['conc_sLES']['low']:
            causes.append("Concentración baja - Revisar proporciones de mezcla")
        
        return causes

# Analizar causas raíz
analyzer = RootCauseAnalyzer()
clf, feature_importance = analyzer.train_cause_classifier(process_data_with_stability)

print("Importancia de características para detección de anomalías:")
print(feature_importance)

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

class ProcessDashboard:
    def __init__(self, process_data):
        self.df = process_data
        self.features = detector.features
        
    def create_main_dashboard(self):
        """Crear dashboard principal del proceso"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Temperatura Reactor', 'Flujo EO', 
                          'pH Neutralización', 'Concentración SLES',
                          'Estabilidad del Proceso', 'Anomalías Detectadas')
        )
        
        # Gráfico de temperatura
        fig.add_trace(
            go.Scatter(x=self.df['timestamp'], y=self.df['temp_reactor'], 
                      name='Temperatura', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Gráfico de flujo EO
        fig.add_trace(
            go.Scatter(x=self.df['timestamp'], y=self.df['flujo_EO'], 
                      name='Flujo EO', line=dict(color='green')),
            row=1, col=2
        )
        
        # Gráfico de pH
        fig.add_trace(
            go.Scatter(x=self.df['timestamp'], y=self.df['ph_neutralizacion'], 
                      name='pH', line=dict(color='red')),
            row=2, col=1
        )
        
        # Gráfico de concentración
        fig.add_trace(
            go.Scatter(x=self.df['timestamp'], y=self.df['conc_sLES'], 
                      name='Conc. SLES', line=dict(color='purple')),
            row=2, col=2
        )
        
        # Gráfico de estabilidad
        fig.add_trace(
            go.Scatter(x=self.df['timestamp'], y=self.df['process_stability'], 
                      name='Estabilidad', line=dict(color='orange')),
            row=3, col=1
        )
        
        # Gráfico de anomalías
        anomalies = self.df[self.df['is_anomaly']]
        fig.add_trace(
            go.Scatter(x=anomalies['timestamp'], y=anomalies['anomaly_score'], 
                      mode='markers', name='Anomalías', marker=dict(color='red', size=8)),
            row=3, col=2
        )
        
        fig.update_layout(height=800, title_text="Monitorización Proceso SLES")
        return fig
    
    def create_risk_matrix(self):
        """Crear matriz de riesgo"""
        risk_data = []
        
        for _, row in self.df[self.df['is_anomaly']].iterrows():
            severity = self._calculate_severity(row)
            probability = self._calculate_probability(row)
            
            risk_data.append({
                'timestamp': row['timestamp'],
                'severity': severity,
                'probability': probability,
                'risk_level': severity * probability
            })
        
        return pd.DataFrame(risk_data)
    
    def _calculate_severity(self, row):
        """Calcular severidad basada en desviaciones"""
        severity = 0
        
        if row['temp_reactor'] > 43: severity += 2
        if row['flujo_EO'] > 260: severity += 2
        if row['ph_neutralizacion'] < 6.2: severity += 1
        if abs(row['conc_sLES'] - 70) > 3: severity += 1
        
        return min(severity, 5)
    
    def _calculate_probability(self, row):
        """Calcular probabilidad basada en duración y magnitud"""
        return min(3 + (row['anomaly_score'] * 2), 5)

# Crear dashboard
dashboard = ProcessDashboard(process_data_with_stability)
fig = dashboard.create_main_dashboard()
risk_matrix = dashboard.create_risk_matrix()

print("Matriz de riesgo calculada:")
print(risk_matrix.head())

class AuditSystem:
    def __init__(self, process_data, hazop_table):
        self.process_data = process_data
        self.hazop_table = hazop_table
        self.audit_log = []
        
    def log_anomaly_event(self, timestamp, anomaly_data, actions_taken):
        """Registrar evento de anomalía"""
        log_entry = {
            'event_id': f"ANOM_{timestamp.strftime('%Y%m%d_%H%M%S')}",
            'timestamp': timestamp,
            'anomaly_type': anomaly_data['anomaly_type'],
            'parameters': anomaly_data['critical_parameters'],
            'identified_causes': anomaly_data['possible_causes'],
            'actions_taken': actions_taken,
            'risk_assessment': self._assess_risk(anomaly_data),
            'follow_up_required': len(anomaly_data['possible_causes']) > 0,
            'audit_trail': self._create_audit_trail(anomaly_data)
        }
        
        self.audit_log.append(log_entry)
        return log_entry
    
    def generate_pha_report(self, start_date, end_date):
        """Generar reporte PHA (Process Hazard Analysis)"""
        period_data = self.process_data[
            (self.process_data['timestamp'] >= start_date) & 
            (self.process_data['timestamp'] <= end_date)
        ]
        
        anomalies = period_data[period_data['is_anomaly']]
        
        report = {
            'period': f"{start_date} to {end_date}",
            'total_anomalies': len(anomalies),
            'risk_distribution': self._calculate_risk_distribution(anomalies),
            'top_deviation_types': self._get_top_deviations(anomalies),
            'effectiveness_safeguards': self._assess_safeguards(anomalies),
            'recommendations': self._generate_recommendations(anomalies)
        }
        
        return report
    
    def _assess_risk(self, anomaly_data):
        """Evaluar riesgo del evento"""
        return "Alto" if any('temperatura' in cause.lower() for cause in anomaly_data['possible_causes']) else "Medio"
    
    def _create_audit_trail(self, anomaly_data):
        """Crear trail de auditoría"""
        return {
            'detection_time': datetime.now(),
            'analyzed_by': 'AI_System',
            'hazop_references': self._link_to_hazop(anomaly_data),
            'data_sources': detector.features,
            'model_confidence': 0.85
        }
    
    def _link_to_hazop(self, anomaly_data):
        """Enlazar con análisis HAZOP relevante"""
        relevant_hazop = []
        for cause in anomaly_data['possible_causes']:
            if 'válvula' in cause.lower():
                relevant_hazop.append('FO-101 Maintenance Procedure')
            if 'temperatura' in cause.lower():
                relevant_hazop.append('TISA-301 Safety System')
        return relevant_hazop

# Sistema de auditoría
audit_system = AuditSystem(process_data_with_stability, hazop_table)

# Simular algunos eventos de auditoría
anomaly_events = process_data_with_stability[process_data_with_stability['is_anomaly']].head(3)
for _, event in anomaly_events.iterrows():
    cause_analysis = analyzer.generate_root_cause_analysis(
        process_data_with_stability, 
        [event.name]
    )
    
    if not cause_analysis.empty:
        audit_entry = audit_system.log_anomaly_event(
            event['timestamp'],
            cause_analysis.iloc[0].to_dict(),
            ['Parada automática iniciada', 'Notificado equipo de mantenimiento']
        )

print("Sistema de auditoría configurado")
print(f"Eventos registrados: {len(audit_system.audit_log)}")

class SLESRiskManagementSystem:
    def __init__(self):
        self.hazop_analyzer = HAZOPAnalizer()
        self.data_simulator = ProcessDataSimulator()
        self.anomaly_detector = AnomalyDetector()
        self.root_cause_analyzer = RootCauseAnalyzer()
        self.audit_system = None
        self.dashboard = None
        
    def initialize_system(self):
        """Inicializar sistema completo"""
        print("Inicializando Sistema de Gestión de Riesgos SLES...")
        
        # 1. Análisis HAZOP
        hazop_table = self.hazop_analyzer.generate_hazop_table()
        print("✓ Análisis HAZOP completado")
        
        # 2. Datos del proceso
        process_data = self.data_simulator.simulate_normal_operation()
        process_data_with_anomalies = self.data_simulator.inject_anomalies(process_data)
        print("✓ Datos del proceso generados")
        
        # 3. Detección de anomalías
        processed_data = self.anomaly_detector.detect_anomalies(process_data_with_anomalies)
        processed_data = self.anomaly_detector.calculate_process_stability_index(processed_data)
        print("✓ Sistema de detección de anomalías configurado")
        
        # 4. Análisis de causas raíz
        clf, feature_importance = self.root_cause_analyzer.train_cause_classifier(processed_data)
        print("✓ Analizador de causas raíz entrenado")
        
        # 5. Sistema de auditoría
        self.audit_system = AuditSystem(processed_data, hazop_table)
        print("✓ Sistema de auditoría configurado")
        
        # 6. Dashboard
        self.dashboard = ProcessDashboard(processed_data)
        print("✓ Dashboard inicializado")
        
        return processed_data, hazop_table
    
    def run_real_time_monitoring(self, live_data):
        """Ejecutar monitoreo en tiempo real"""
        results = {
            'current_status': 'NORMAL',
            'anomalies_detected': 0,
            'risk_level': 'BAJO',
            'recommendations': []
        }
        
        # Detectar anomalías
        live_processed = self.anomaly_detector.detect_anomalies(live_data)
        
        if live_processed['is_anomaly'].any():
            results['current_status'] = 'ALERTA'
            results['anomalies_detected'] = live_processed['is_anomaly'].sum()
            results['risk_level'] = 'MEDIO'
            
            # Analizar causas
            anomaly_indices = live_processed[live_processed['is_anomaly']].index
            causes = self.root_cause_analyzer.generate_root_cause_analysis(
                live_processed, anomaly_indices
            )
            
            results['recommendations'] = causes['recommended_actions'].tolist()
        
        return results

# Ejecutar sistema completo
risk_system = SLESRiskManagementSystem()
processed_data, hazop_table = risk_system.initialize_system()

print("\n" + "="*60)
print("SISTEMA DE GESTIÓN DE RIESGOS SLES IMPLEMENTADO EXITOSAMENTE")
print("="*60)
print(f"• Análisis HAZOP: {len(hazop_table)} escenarios de riesgo")
print(f"• Datos procesados: {len(processed_data)} muestras")
print(f"• Anomalías detectadas: {processed_data['is_anomaly'].sum()} eventos")
print(f"• Sistema de auditoría: {len(risk_system.audit_system.audit_log)} eventos registrados")

# Generar reporte final
def generate_final_report(processed_data, hazop_table, audit_system):
    """Generar reporte ejecutivo final CORREGIDO"""
    
    # Calcular métricas de clasificación correctamente
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    # Datos reales vs predicciones
    y_true = (processed_data['anomaly'] > 0).astype(int)  # Anomalías reales (inyectadas)
    y_pred = processed_data['is_anomaly'].astype(int)     # Anomalías detectadas
    
    # Calcular métricas
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, zero_division=0) * 100
    recall = recall_score(y_true, y_pred, zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100
    cm = confusion_matrix(y_true, y_pred)
    
    # Contar falsos positivos CORRECTAMENTE
    false_positives = ((y_pred == 1) & (y_true == 0)).sum()
    
    report = {
        'executive_summary': {
            'total_analysis_period': f"{processed_data['timestamp'].min()} to {processed_data['timestamp'].max()}",
            'process_availability': f"{(1 - processed_data['is_anomaly'].mean()) * 100:.2f}%",
            'risk_reduction_potential': "Estimado 40-60% mediante detección temprana",
            'key_risk_areas': ['Sulfatación EO', 'Control pH Neutralización', 'Estabilidad Agitación']
        },
        'performance_metrics': {
            'total_samples': len(processed_data),
            'real_anomalies': int(y_true.sum()),
            'detected_anomalies': int(y_pred.sum()),
            'false_positives': int(false_positives),
            'false_negatives': int(cm[1, 0]),  # Anomalías reales no detectadas
            'accuracy': f"{accuracy:.1f}%",
            'precision': f"{precision:.1f}%",  # De lo que detectó como anomalías, cuántas realmente lo eran
            'recall': f"{recall:.1f}%",        # De las anomalías reales, cuántas detectó
            'f1_score': f"{f1:.1f}%",
            'mean_time_to_detect': "15-30 minutos estimados"
        },
        'recommendations': [
            'Implementar mantenimiento predictivo en válvulas EO',
            'Mejorar calibración de pH-metros (semanal vs mensual)',
            'Agregar sensores de vibración en agitadores',
            'Automatizar paradas de seguridad basadas en ML',
            f'Ajustar modelo de detección: {false_positives} falsas alarmas detectadas'
        ]
    }
    
    # Agregar diagnóstico del modelo
    if false_positives > 0.1 * y_pred.sum():  # Si más del 10% son falsas alarmas
        report['performance_metrics']['diagnostic'] = "ALERTA: Demasiados falsos positivos - ajustar sensibilidad del modelo"
    elif recall < 70:
        report['performance_metrics']['diagnostic'] = "ALERTA: Muchas anomalías no detectadas - mejorar sensibilidad"
    else:
        report['performance_metrics']['diagnostic'] = "Desempeño del modelo dentro de parámetros aceptables"
    
    return report

# Generar y mostrar reporte final
final_report = generate_final_report(processed_data, hazop_table, risk_system.audit_system)
print("\nREPORTE EJECUTIVO FINAL:")
print("="*50)
for section, content in final_report.items():
    print(f"\n{section.upper().replace('_', ' ')}:")
    if isinstance(content, dict):
        for k, v in content.items():
            print(f"  • {k}: {v}")
    else:
        for item in content:
            print(f"  • {item}")

# 1. Crear la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# 2. Definir el layout de la aplicación
app.layout = dbc.Container([
    html.H1("Dashboard de Monitoreo del Proceso SLES"),
    dcc.Graph(id='main-dashboard', figure=fig),
    # Puedes agregar más componentes aquí
])

# 3. EJECUTAR EL SERVIDOR CON EL MÉTODO CORRECTO
if __name__ == '__main__':
    app.run(debug=True)
