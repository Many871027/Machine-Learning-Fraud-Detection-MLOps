# 📄 Reporte Integral del Ciclo de Vida MLOps: Sistema de Prevención de Fraude Crediticio

Este documento asienta formalmente el historial de diseño, arquitectura técnica y las decisiones de reestructuración computacional empleadas para transformar el antiguo experimento de regresión local (Telco Churn) en una pasarela API de Producción de clase mundial robusta a altas frecuencias de inferencia.

---

## FASE 1: Adquisición y Gobierno de Datos (Data Engineering)
Ante el requerimiento de detectar transacciones espurias, inyectamos el ecosistema transaccional a nivel de Big Data. Implementamos un micro-script (`download_data.py`) utilizando la API de `kagglehub` que asegura un ingreso determinístico de un volumen masivo (150 MB constantes) bajo el esquema "Credit Card Fraud".

> [!NOTE]
> **Privacidad PCA:** El dataset de Fraude ingresado ya contempla anonimización financiera a través de un Análisis de Componentes Principales (Características V1 hasta V28), protegiendo nativamente la Identificación de Personas Físicas (PII).

---

## FASE 2: Entorno Modular de Procesamiento de Datos (ETL)
El módulo original heredado contenía procesamientos de texto, casillas nulas y transformaciones sintácticas. Re-escribimos por completo el módulo `src/data_pipeline.py`.
Dado que los Vectores PCA conforman arreglos unidimensionales puros (flotantes), **deprecamos toda clase de `LabelEncoder`** y el mapeo `NaN`, aligerando la carga I/O drásticamente.

---

## FASE 3: Definición Arquitectónica (Las Eliminaciones Estratégicas)
> [!IMPORTANT]
> **Justificación de Descarte Técnico: Por qué NO utilizamos todos los algoritmos del Notebook original.**

El escalonamiento de un dataset local universitario (7,000 registros) hacia parámetros reales (285,000 registros x 30 componentes paralelos) cambia paradigmáticamente las leyes físicas del *Machine Learning*. Se decidió explícitamente **NO implementar** los siguientes modelos por tres severidades MLOps:

1. **KNN (Clasificación Espacial):** Este algoritmo no se serializa realmente. Mantiene en RAM su memoria espacial histórica. Para inferir transacciones fraudulentas en tiempo real a través de API, calcular distancias euclidianas contra 285k registros en el backend causaría caídas por latencia inaceptable para pasarelas (Timeouts mayores a 3 segundos).
2. **SVM (Máquinas Vectores de Soporte):** Su matriz de kernel ostenta una complejidad algorítmica y de entrenamiento de $O(N^2)$ a $O(N^3)$. Sumado al entorno de validación cruzada y sintéticos (SMOTE), su demanda requería clusters de Cloud Computing masivos ajenos al Hardware local del *Host*, conduciendo indefectiblemente a un *Out Of Memory Error*.
3. **MultiLayer Perceptron (Redes Neuronales de Densidad Lineal):** Para conjuntos de datos 100% numéricos-tabulares unicolumna, los modelos probabilísticos basados en ensamble de árboles Gradient Boosting destruyen virtual y teóricamente a perceptrones biológicos. 
4. **Voting (Meta-Estiamción Ciclomática):** Agregar un "votante" obligaría ejecutar simultáneamente en memoria 3 procesos Predict para enviar UNA sola respuesta HTTP (Disminuye drásticamente el *throughput* / Carga máxima que soporta el microservicio).
5. **Linear Regression:** Prohibido teóricamente dado que esto es una topología discreta de Clasificación Binaria estricta (Fraudes vs Normal), no una predicción contínua de cantidades numéricas financieras infinitas.

---

## FASE 4: Flujo Experimental y Seguimiento Telemetrico (MLflow)
Corrimos en frío el *script* orquestador: `run_experiments.py`. 
Para contrarrestar la proporción agresiva contra el caso criminal real (0.17% son fraudes reales / 99.8% legítimos), activamos el sub-circuito **SMOTE** balanceando a pesos probabilísticos ideales.
El registro telemétrico (*MLruns Tracker*) observó la contienda cíclica bajo validación cruzada utilizando exclusivamente Algoritmos viables en producción como: `NaiveBayes`, `DecisionTree`, `RandomForest` y `XGBoost`. 

---

## FASE 5: Compilación Universal a Producción (Model Deployment)
Basados en la exploración, se re-compiló el binario productivo a través de `src/model_pipeline.py`.
- **Estrategia MVP:** Retiramos la sobresaturación artificial SMOTE para permitir que el algoritmo `XGBClassifier` leyese los patrones intrínsecos al natural en `fit()`. 
- **Persistencia de Serialización:** Se generaron el empaquetado del escalador universal estadístico `artifacts/scaler.pkl` y los metadatos dinámicos nativos en `artifacts/model`, logrando atrapar y certificar una **Exactitud del 99.94%** sobre un AUC del **0.9390**.

---

## FASE 6: Entrega Backend y Endpoints Analíticos (FastAPI Server)
La arquitectura consumible fue totalmente levantada en `app/main.py`.
- Enrutamos y reescribimos el esquema de blindaje *Pydantic*, transformándolo en una super-estructura flotante denominada `class TransactionFeatures(BaseModel)`. Previene inyecciones basura al predecir a base de *typehints* financieros flotantes (Las variables V1-V28, Monto y Tiempo).
- Activamos e integramos la respuesta de monitoreo corporativo en un endpoint GET en texto plano sobre `/metrics`.

---

## FASE 7: Operaciones Computacionales y Despliegue Automático (CI/CD)
En lugar de finalizar en la simple API, transformamos los activos en un ambiente escalable para múltiples ingenieros.
1. Se programó el ecosistema virtual `notebooks/fraud-prediction.ipynb` limitando la experimentación visual caótica fuera de ramas de despliegue.
2. Se inyectaron **Pruebas de Test** rígidas y unitarias en `tests/test_pipeline.py`. Ningún modelo sube a la Nube si los endpoints y los artefactos devuelven código HTTP 500.
3. Se integró una Orquestación **GitHub Action Pipelines** (`.github/workflows/ci_cd.yml`) amontonando los tests y dictando reglas continuas para ramas `main`, lo que cierra de forma circular todo un espectro empresarial de control desde MLOps Nivel 0 (Manual) al robusto escalón de MLOps Nivel 1.
