# 📄 Reporte de Arquitectura MLOps: Predicción de Fraude Crediticio

**Índice Arquitectónico Estructurado (Mapeo de Distribución)**
| Formulación del Problema: 1. Introducción ----------------------------------------------------Pág. 3
| Pipeline DataOps: 2. Carga y Preprocesamiento de Datos --------------------------Pág. 4 
| Diseño Experimental: 3. Metodología de Experimentación --------------------------Pág. 5 
| Benchmarking Algorítmico: 4. Análisis Comparativo de Modelos ----------------Pág. 7 
| Telemetría de Decisión: 5. Resultados y Evaluación del Modelo Óptimo --------Pág. 9 
| Observabilidad: 6. Interpretabilidad de Variables —--.--------------------------------Pág. 12 
| Operacionalización: 7. Arquitectura de Despliegue MLOps—--------------------.-Pág. 14 
| Cierre Analítico: 8. Conclusiones y Trabajo Futuro—---------------------------------Pág. 16 
| Fundamentación: 9. Referencias Bibliográficas—--------------------------------------Pág. 18 

---

## 1. Introducción
El análisis forense y predictivo de transacciones fraudulentas constituye uno de los desafíos computacionales más hostiles en el modelado de clasificación binaria bajo aprendizaje supervisado. En el ecosistema financiero, la detección estocástica algorítmica impacta directamente la asimetría entre las pérdidas por contracargos y la experiencia del usuario final (falsos positivos/declinación de pagos lícitos).

En este proyecto, se diseña una arquitectura MLOps integral para procesar el denso espacio vectorial de *Credit Card Fraud*. A diferencia de conjuntos tradicionales, el contexto del fraude conlleva un hiper-desbalance estructural de clases (0.17% positivos). La maximización de la capacidad de generalización algorítmica exigió podar estimadores paramétricos de costo exponencial (SVM, KNN) y transicionar estrictamente a ecosistemas de ensamble (*Gradient Boosting, Random Forest*).

---

## 2. Carga y Preprocesamiento de Datos (Pipeline DataOps)
La calidad y la velocidad de inferencia de la pasarela API dependen estrechamente del costo computacional de Entrada/Salida (I/O). El módulo `src/data_pipeline.py` fue refactorizado para operar bajo los principios de Data Engineering de alta eficiencia.

* **Anonimización y Privacidad Nativa (PCA):** El ecosistema transaccional masivo (~150 MB) procesado asienta sus características bajo un Análisis de Componentes Principales. Al estar compuesto por 28 vectores ortogonales (`V1` a `V28`) además de las características de `Monto` y `Tiempo`, se protege nativamente la Identificación de Personas Físicas.
* **Depuración de Transformadores:** Dado que los vectores PCA conforman un arreglo unidimensional puro de valores de punto flotante, se deprecó el uso de `LabelEncoder` y el mapeo de transformaciones de texto nulos.

---

## 3. Metodología de Experimentación
Para blindar nuestras evaluaciones frente a la "maldición de la varianza" y generalizar sobre una matriz inmensa de 284,807 registros, estandarizamos técnicas paramétricas estrictas.

**3.1. Validación Cruzada (Cross-Validation)**
Se implementó `StratifiedKFold(n_splits=3)` junto al `GridSearchCV`.
- **Estratificación Severa:** Asegura que la subrepresentación atípica del 0.17% (Fraude=1) se sostenga geométrica y equitativamente dividida entre los bloques de entrenamiento (`X_train`) y test (`X_test`).
- **Inclusión Preventiva del Ecosistema de Producción:** Para validar científicamente nuestra hipótesis, se inyectó un iterador nativo de `XGBoost_Production_Nativo` directamente en el código base de experimentación, el cual correría en paralelo bajo validación cruzada.

**3.2. Abordaje del Desbalance de Clases**
El tratamiento requirió dos aproximaciones diametralmente distintas.
- **Remuestreo Espacial Sintético (SMOTE):** Probado sintéticamente durante `run_experiments.py` para Random Forest, Decision Tree y Bayes. 
- **Aprendizaje Puro sin Aditivos (Production):** Demostrado empíricamente en `XGBoost_Production_Nativo`, omitiendo métodos sintéticos que alteren la hiperestructura original y apostando por la entropía matemática pura de asimetría.

---

## 4. Análisis Comparativo de Modelos
El escrutinio telemétrico proveniente de MLflow consolidó la jerarquía de predicción demostrando la cruda fisura entre entornos adulterados y modelos nativos estables listos para negocio:

| Modelo Experimental | Ajuste de Entorno | Accuracy Óptimo | AUC-ROC | F1-Score | Precision | Recall | Decisión de Viabilidad |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🚀 **XGBoost_Production_Nativo** | **Sin SMOTE (Puro)** | **99.94%** | 0.9390 | **0.8298** | **0.8667** | 0.7959 | **Viable Operativamente** |
| **RandomForest** | **CV + SMOTE** | 99.32% | **0.9746** | 0.3096 | 0.1875 | 0.8878 | Inviable (Demasiados Falsos Positivos) |
| **XGBoost** | CV + SMOTE | 99.07% | 0.9675 | 0.2417 | 0.1407 | 0.8571 | Inviable |
| **DecisionTree** | CV + SMOTE | 95.70% | 0.9498 | 0.0678 | 0.0352 | **0.9082** | Declinado Absoluto |

---

## 5. Resultados y Evaluación del Modelo Óptimo (Decisión de Negocio)
La elección final transita inexorablemente al abandono de las lógicas SMOTE cruzadas, declarando como triunfador inobjetable a **XGBoost_Production_Nativo**.

* **Precisión Monetaria y Riesgo Operacional:** Si implementáramos localmente el algoritmo `RandomForest (CV+SMOTE)` respaldados ciegamente por su monumental `AUC 0.97`, caeríamos en un suicidio empresarial: Retiene un raquítico **18% de Precision**. En la vida real, el sistema de la pasarela bancaria bloquearía legítimamente docenas de tarjetas sanas por error (Falsos Positivos), multiplicando un abandono colosal de clientes legítimos frustrados e indignando soporte telefónico.
* **El Brillante Balance Nativo:** El algoritmo **`XGBoost_Production_Nativo`** evitó las distorsiones, logrando **86.67% de Precision** con un robusto 80% de recall sistémico. Significa que de cada 10 tarjetas que el orquestador rechaza por "Fraude", prácticamente 9 son estafas internacionalmente comprobadas; validando la hipótesis de que, sobre hiper-datos ($N \approx 300,000$), la manipulación excesiva de SMOTE quiebra las dinámicas locales del clasificador.

---

## 6. Interpretabilidad de Variables (Feature Importance)
La opacidad inherente tras el tamizado de Seguridad por Análisis de Componentes Principales (V1-V28) dificulta extraer nombres explícitos corporativos, pero consolida un modelo de caja de cristal matemática.
- `Amount` (Monto Neto de la Transacción) y `Tiempo` emergen como discriminantes estocásticos. En montos exóticos asimétricamente altos a media noche, la densidad del vector predictivo sube abismalmente frente al control base de consumo ordinario.

---

## 7. Arquitectura de Despliegue (MLOps)
El *Lifecycle* culmina encapsulando ingeniería en repositorios estables:
- **Serialización Automática (MLflow):** Persistencia de binarios `/artifacts/model` compilados sin sobrecargar el *training-serving skew*.
- **Endpoint Asíncrono RestFul:** FastApi orquestando una estructura `TransactionFeatures(BaseModel)` con 30 campos matemáticos pre-parseados a *float*.
- **Integración Continua (GitHub Actions CI/CD):** Correlato de despliegue mediante instancias unitarias automatizadas (`pytest tests/`) para sellar a `main` desde `.github/workflows`.

---

## 8. Conclusiones
Hemos escalado una premisa empírica universitaria hacia pasarelas API con normativas rígidas industriales de seguridad. 

El modelo ganador (**XGBoost_Production_Nativo**) no solo suplantó algoritmos costosos como *KNN* y dependientes como *Lineales*, sino que demostró categóricamente (al haber superado las fases integradas del comparativo `run_experiments.py`) que **el balanceo de SMOTE sobre data financiera ultradesbalanceada es nocivo financieramente por las penalizaciones de Falso Positivo**. Con un rendimiento estabilizado del 82.9% F1 y 99.94% de exactitud corporativa, el contenedor desplegado resalta la madurez superlativa lograda dentro de la arquitectura MLOps.

---

## 9. Referencias
- Chawla, N. V. (2002). SMOTE: Synthetic Minority Over-sampling Technique. (Para contrastar cómo la interpolación sintética castigó el P-R Curve en transacciones masivas).
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. (Sustento absoluto de la eficiencia algorítmica).
- Raj, E. (2021). Engineering MLOps: Rapidly build, test, and manage production-ready machine learning life cycles at scale. Packt. (Gobernanza CI/CD y despliegue preventivo).
- Aggarwal, C. C. (2024). Probability and Statistics for Machine Learning: A Textbook. Springer. (Validación Bayesiana e integración cruzada de experimentos estocásticos hiperdimensionales).
