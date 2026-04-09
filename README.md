# Crédit Card Fraud Detection MLOps

Arquitectura MLOps de Grado Empresarial orientada a la detección predictiva interbancaria de transacciones fraudulentas usando el poderoso algoritmo **XGBoost**. 

Construida con Python, MLflow y FastAPI, esta pasarela implementa estrictas prácticas analíticas nativas superando el hiper-desbalance estructural de clases (0.17%) sin incurrir en penalizaciones financieras provocadas por métodos académicos distorsionados como SMOTE en el entorno Productivo.

##  Prerequisitos
- Python 3.10+
- Git

##  Instalación y Configuración

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/Many871027/Machine-Learning-Fraud-Detection-MLOps.git
   cd Machine-Learning-Fraud-Detection-MLOps
   ```

2. **Crear e inicializar un entorno virtual (Recomendado):**
   ```bash
   python -m venv venv
   # En Windows (Powershell):
   .\venv\Scripts\activate
   # En Linux/Git Bash:
   source venv/bin/activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

##  Fases de Ejecución Operativa

### 1. Ingestión de Datos Estocásticos (DataOps)
El *Credit Card Fraud Dataset* nativo no se incluye al clonar por su peso computacional (más de 150MB). Ejecuta el proceso de ingesta (soportado por KaggleHub) para traerlo limpiamente a tu entorno local:
```bash
python download_data.py
```
*(Asegúrate de comprobar que el archivo descargado `creditcard.csv` resida en la raíz del proyecto para uso local)*.

### 2. Ecosistema de Experimentación Cruzada (MLflow)
Ejecutar la suite telemétrica para comparar el impacto destructivo del balanceo sintético en contraste con algoritmos nativos frente al riesgo de *Falso Positivo*:
```bash
python run_experiments.py
```
>  *Esto forzará a Random Forest, Decision Tree y Bayes a competir contra métricas cruzadas, guardando silenciosa y deterministamente toda la telemetría resultante en el entorno `/mlruns/`.*

### 3. Compilación Serializada del Campeón Producción (XGBoost)
Ejecutar el conducto binario enfocado puramente en el encapsulamiento para servicio (Servidor). Extrae la versión hiper-optimizada de *XGBoost_Production_Nativo* y su Escalamiento Numérico para generar memoria asíncrona:
```bash
python -m src.model_pipeline
```
>  *Generará los binarios empaquetados (`model/` estructurado por mlflow) y (`scaler.pkl`) en la carpeta productiva `artifacts/`.*

## 📡 Despliegue en Nube y Servidor API Restful

Tras la compilación exitosa, levanta la compuerta de backend para poder recibir los Puntos Flotantes del servidor:
```bash
uvicorn app.main:app --reload
```
Abre tu navegador de preferencia en **http://127.0.0.1:8000/docs** para acceder al portal *Swagger UI*.

###  Inferencia Predictiva y Observabilidad Empresarial

Una vez dentro de la interfaz gráfica del Swagger:
*   **Ataque de Pago (`POST /predict`):** Este Endpoint ejecuta simulación de compra real. Transmite el JSON de valores hacia la estructura hermética `TransactionFeatures(BaseModel)`. FastAPI correrá la validación vectorial al segundo, devolviendo la probabilidad binaria del Riesgo Operacional (`1`: Fraude Detectado).
*   **Telemetría Comparativa (`GET /metrics`):** Este endpoint hace proxy interno con el servidor central de análisis cruzado (*MLflow*), exportando en un Dashboard transaccional (JSON) todo el comparativo entre experimentos. Podrás constatar en tiempo real cómo RandomForest con CV+SMOTE acierta apenas el 18.7% de *Precision* empresarial; contrastando empíricamente la supremacía dorada del 86.6% del entorno XGBoost Estructural.

---
**Automatizado y Sellado bajo CI/CD (GitHub Actions 🛡️)**
