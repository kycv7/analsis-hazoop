# ============================================================
# REQUERIMIENTOS PARA SISTEMA DE GESTIÓN DE RIESGOS SLES
# Integración HAZOP + Machine Learning + Análisis de Procesos
# ============================================================

# ------------------------------------------------------------
# DEPENDENCIAS PRINCIPALES
# ------------------------------------------------------------

# Framework de ciencia de datos y machine learning
scikit-learn==1.3.2
pandas==2.1.1
numpy==1.25.2
scipy==1.11.2

# Visualización y dashboards
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0
dash==2.14.1
dash-bootstrap-components==1.4.0

# Explicabilidad de modelos ML
shap==0.42.1

# Procesamiento de series temporales
statsmodels==0.14.0

# ------------------------------------------------------------
# DEPENDENCIAS DE APLICACIÓN WEB/DASHBOARD
# ------------------------------------------------------------

# Framework web
flask==2.3.3

# Componentes de dashboard interactivo
dash-core-components==2.0.0
dash-html-components==2.0.0
dash-table==5.0.0

# Autenticación y seguridad
dash-auth==2.0.0
flask-login==0.6.2
werkzeug==2.3.7

# ------------------------------------------------------------
# PROCESAMIENTO DE DATOS Y ALMACENAMIENTO
# ------------------------------------------------------------

# Conexión a bases de datos y almacenamiento
sqlalchemy==2.0.20
psycopg2-binary==2.9.7  # Para PostgreSQL
pymysql==1.1.0  # Para MySQL
redis==5.0.1

# Manipulación de datos temporales
python-dateutil==2.8.2
pytz==2023.3

# Formatos de archivo
openpyxl==3.1.2
xlrd==2.0.1

# ------------------------------------------------------------
# UTILIDADES Y HERRAMIENTAS
# ------------------------------------------------------------

# Logging y monitoreo
loguru==0.7.0
sentry-sdk==1.30.0

# Utilidades del sistema
python-dotenv==1.0.0
pyyaml==6.0.1
requests==2.31.0

# Testing y validación
pytest==7.4.2
pytest-cov==4.1.0
hypothesis==6.82.0

# ------------------------------------------------------------
# OPTIMIZACIÓN Y DESPLIEGUE
# ------------------------------------------------------------

# Aceleración numérica (opcional)
numba==0.58.1

# Cache para mejora de rendimiento
joblib==1.3.2

# Entorno virtual (desarrollo)
virtualenv==20.24.3

# Empaquetamiento
setuptools==68.2.2
wheel==0.41.2

# ------------------------------------------------------------
# SEGURIDAD Y CUMPLIMIENTO
# ------------------------------------------------------------

# Cifrado y seguridad de datos
cryptography==41.0.4

# Auditoría y trazabilidad
python-json-logger==2.0.7

# ------------------------------------------------------------
# DEPENDENCIAS DE DOCUMENTACIÓN
# ------------------------------------------------------------

# Generación de reportes y documentación
jupyter==1.0.0
notebook==7.0.4
markdown==3.4.4
sphinx==7.2.6

# ------------------------------------------------------------
# ESPECIFICACIONES DEL SISTEMA
# ------------------------------------------------------------

# Versión de Python requerida
# Python >= 3.9, < 3.12

# Sistema operativo: Multiplataforma (Windows/Linux/macOS)

# Memoria recomendada: 8GB RAM mínimo, 16GB recomendado

# Almacenamiento: 500MB para instalación + espacio para datos
