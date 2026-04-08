# Documentación Técnica: Plataforma MLOps de Clasificación

Esta documentación aborda la arquitectura técnica del proyecto y establece la guía oficial paso a paso para **clonar y reproducir** este ecosistema de *Machine Learning* utilizando un conjunto de datos (dataset) completamente distinto.

---

## 1. Arquitectura del Proyecto

El proyecto ha sido refactorizado desde un Jupyter Notebook hasta alcanzar una arquitectura de microservicios locales (MLOps). El código se estructura de la siguiente manera:

```text
├── app/
│   ├── main.py                # Servidor FastAPI (Endpoints POST /predict y GET /metrics)
│   └── __init__.py
├── artifacts/                 # Contiene el modelo final (.pkl/MLmodel) y escaladores tras compilar
├── notebooks/
│   └── churn-prediction.ipynb # Entorno de Experimentación (Benchmark de 6 modelos)
├── src/
│   ├── config.py              # Diccionarios de hiperparámetros (GridSearchCV, SMOTE)
│   ├── data_pipeline.py       # Lógica ETL (Limpieza, One-Hot Encoding, imputación)
│   └── model_pipeline.py      # Entrenador nativo y conexión central con MLflow
├── tests/
│   └── test_pipeline.py       # CI/CD: Pruebas unitarias de Github Actions (pytest)
├── mlruns/                    # Base de datos local SQLite y tracking de MLflow
├── run_experiments.py         # Script silencioso para ejecutar la validación completa cruzada
└── requirements.txt           # Dependencias requeridas
```

---

## 2. Instalación de Entorno

> [!IMPORTANT]
> Se asume que cuentas con un entorno Windows y Python 3.10+. Evita empaquetar de forma global (usa entornos virtuales).

1. **Crear e inicializar el entorno virtual**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
2. **Instalar dependencias clave**:
   ```powershell
   pip install -r requirements.txt
   ```

---

## 3. Guía de Reproducibilidad (Adaptando Nuevos Datos)

Si deseas utilizar esta infraestructura predictiva en otro problema de negocio (por ejemplo: Riesgo de Crédito, Predicción de Enfermedades o Detección de Fraude), debes seguir estos **5 pasos secuenciales**:

### Paso 1: Introducir el Nuevo Dataset
Coloca tu nuevo archivo tabular (ej. `nuevo_dataset.csv`) en la raíz del proyecto. Identifica claramente cuál será tu columna clase predictiva (Target Variable), asegurándote que sea un problema binario `[0, 1]` o transformable a ello.

### Paso 2: Parametrizar el *Data Pipeline*
Abre el archivo `src/data_pipeline.py`. 
- Localiza la función `load_and_preprocess_data`.
- Reemplaza el tratamiento estático de columnas (ej. las imputaciones de "TotalCharges" o las columnas eliminadas como "customerID" que correspondían a Telco).
- Adapta la técnica del `LabelEncoder` según las columnas categóricas de tu nuevo archivo.

### Paso 3: Ajustar el Esquema del Backend (FastAPI)
Abre `app/main.py`. Este archivo utiliza *Pydantic* para forzar una validación de seguridad de los JSON que entran a la API.
- Localiza la clase `class CustomerFeatures(BaseModel):`.
- Sustituye todas las variables declaradas allí por **los nombres de las columnas** y **los tipos de datos** (int, float, str) de tu nuevo dataset. Esto es vital para que `/predict` no devuelva un error 422.

### Paso 4: Re-ejecutar el Benchmark de Inferencia (MLflow)
Ve a `run_experiments.py` o a `notebooks/churn-prediction.ipynb`.
1. Cambia la referencia del `.csv` en la función `load_and_preprocess_data()`.
2. Actualiza la variable objetivo: Reemplaza `y = df['Churn']` por tu nueva variable `y = df['Mi_Nueva_Columna_Target']`.
3. Ejecuta el script.
   ```powershell
   .\venv\Scripts\python run_experiments.py
   ```
4. Inicia la interfaz de MLflow: `.\venv\Scripts\mlflow ui` y averigua cuál de las 6 arquitecturas algorítmicas (RandomForest, SVM, XGBoost, etc.) fue la victoriosa.

### Paso 5: Promover a Producción
Por último, entra al archivo `src/model_pipeline.py`.
- Navega abajo hasta la función `train_production_model(X, y)`.
- Reemplaza el algoritmo incrustado ("hardcodeado") por el que haya ganado tu experimento (Ej. Pasa de `XGBClassifier` a `RandomForestClassifier`), alimentando los hiperparámetros campeones que descubriste en MLflow.
- Ejecuta la compilación de producción:
  ```powershell
  .\venv\Scripts\python -m src.model_pipeline
  ```
  > [!TIP]
  > Este comando es necesario porque empaqueta y purga de formal nativa los binarios `artifacts/scaler.pkl` y `artifacts/model` para que FastAPI los pueda consumir instantáneamente con 0 milisegundos de latencia.

### Paso 6: Lanzar a la Web
Levanta el servidor con la integración Uvicorn-FastAPI (sin el `--reload` para evitar errores Socket 10022 de Windows en multithreading):
```powershell
.\venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Dirígete a `http://127.0.0.1:8000/docs` en tu computadora y tu nuevo sistema de Inteligencia Artificial de Producción estará operativo.
