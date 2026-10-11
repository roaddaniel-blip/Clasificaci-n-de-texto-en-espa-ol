# Clasificación de texto en español (Análisis de sentimiento) 

Este notebook contiene la implementación, evaluación y comparación de tres modelos preentrenados del ecosistema de **Hugging Face** para un caso de uso de Procesamiento de Lenguaje Natural (NLP): **Análisis de Sentimiento / Clasificación de Texto** en español.

## Tabla de contenido 

1. **Código del modelo**
   * Configuración de GPU
   * Instalación de dependencias
   * Autenticación en Hugging Face
   * Carga de dataset y preparación de modelos 
   * Pipeline de inferencia y análisis de modelos
   * Medición de latencia
   * Tabla comparativa de resultados de modelos evaluados 

2. **Análisis comparativo de los modelos**
   * Caso de uso
   * Criterios de selección 
   * Tabla comparativa 
   * Análisis de trade-offs 
   * Decisión de ajuste mínimo 
   * Justificación del modelo recomendado 
   * Umbrales y riesgos 

3. **Dataset de soporte**
   * Descripción 
   * Estructura

4. **Consideraciones**
   * Entorno de ejecución 
   * Pasos de ejecución
   * Conclusiones 

5. **Repositorio**
   * Archivos de soporte
   * Referencias 

---
## 1) Código del modelo

* **Configuración de GPU:** Se realizó la validación de GPU en equipo personal.
* **Instalación de dependencias:** Se instalaron las siguientes dependencias:
  * `transformers`: Biblioteca principal de Hugging Face que proporciona las arquitecturas de modelos de lenguaje (como BERT, BETO y RoBERTa) y las herramientas para realizar inferencia mediante pipelines.
  * `datasets`: Utilidad para descargar, estructurar y manipular conjuntos de datos de forma eficiente en formato compatible con modelos de Machine Learning.
  * `evaluate`: Biblioteca de Hugging Face diseñada para calcular métricas de evaluación de modelos de forma estandarizada.
  * `scikit-learn`: Utilizada para calcular métricas estadísticas tradicionales de clasificación (como Accuracy, Precision, Recall y F1-Score) basándose en las predicciones obtenidas.
  * `accelerate`: Herramienta de Hugging Face que optimiza el entrenamiento y la inferencia de modelos, simplificando la ejecución paralela en CPU o GPU.
  * `huggingface_hub`: Cliente oficial para interactuar con la plataforma Hugging Face, permitiendo la descarga de modelos y la autenticación mediante tokens.
* **Autenticación en Hugging Face:** Login interactivo en la plataforma de Hugging Face Hub.
* **Carga de dataset y preparación de modelos:** Se utilizó el dataset `paws-x`, enfocado en la detección de evaluaciones y recomendaciones recibidas por los clientes. Asimismo se crearon y evaluaron tres modelos diferentes de rendimiento, los cuales son:
  * BERT (`bert-base-multilingual-cased` ajustado)
  * BETO (`dccuchile/bert-base-spanish-wwm-cased` ajustado)
  * RoBERTa (`xlm-roberta-base` ajustado)
* **Pipeline de inferencia y análisis de modelos:** Se establece el conjunto ordenado de procesos y se definen las funciones genéricas para evaluar cada uno de los 3 modelos considerando:
  * **a)** Cálculo del tiempo de ejecución (latencia promedio por muestra).
  * **b)** Accuracy.
  * **c)** Precision.
  * **d)** Recall.
  * **e)** F1-Score.
* **Medición de latencia:** Se realiza un comparativo de eficiencia por cada uno de los tres modelos. 
* **Tabla comparativa de resultados de modelos evaluados:** Visualización de los resultados consolidados de precisión, exhaustividad y latencia de inferencia de los tres modelos para facilitar la toma de decisiones técnicas.


## 2) Análisis comparativo de los modelos

* **Caso de uso:** Análisis de Sentimientos y Clasificación de Texto en Español, enfocado específicamente en la detección y análisis de la retroalimentación, opiniones, reseñas y nivel de satisfacción de los clientes respecto a productos o servicios, permitiendo clasificar de forma automatizada las experiencias de usuario en categorías binarias de aceptación o rechazo (**Positivo vs. Negativo**).
* **Criterios de selección:** Para este modelo se utilizaron tres modelos distintos con el objetivo de evaluar de manera óptima los sentimientos en español considerando:
  * Especialización lingüística
  * Capacidad de generalización lingüística
  * Eficiencia de recursos
* **Tabla comparativa:** Visualización de los resultados consolidados de precisión, exhaustividad y latencia de inferencia de los tres modelos para facilitar la toma de decisiones técnicas.

### Análisis de trade-offs

| Modelo | Precisión (F1-Score) | Latencia Promedio (CPU) | Idiomas / Generalización | Tamaño en Disco (Parámetros) | Licencia / Viabilidad Comercial |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RoBERTuito**<br>*(Español / Nativo)* | 1.0000<br>(Perfecto) | ~195.31 ms<br>(La más baja) | Especializado exclusivamente en español (modismos y textos informales). | ~435 MB (110M)<br>– Ideal para arquitecturas económicas o Serverless. | **Permisiva (MIT)**<br>– Apto para uso comercial sin restricciones. |
| **XLM-RoBERTa**<br>*(Multilingual)* | 1.0000<br>(Excelente) | ~375.32 ms<br>(Penalización por multilingüismo) | Global (+100 idiomas); ideal para escalabilidad internacional. | ~1.1 GB (270M)<br>– Duplica los requisitos de RAM/VRAM y eleva costos. | **Permisiva (MIT)**<br>– Apto para uso comercial sin restricciones. |
| **BETO Sentiment** | 0.8889<br>(Menor) | ~427.64 ms<br>(La más alta) | Español (análisis de sentimiento). | No especificado | **CC BY-NC-4.0**<br>– No comercial (Descartado para monetización directa). |

* **Decisión de ajuste mínimo:** Se cargaron los pesos ya especializados de cada modelo (`robertuito-sentiment-analysis`, `beto-sentiment-analysis` y `twitter-xlm-roberta-base-sentiment`) y se aplicó una función de mapeo simple en Python para unificar las etiquetas de salida (`LABEL_2`, `POS`, etc.) a un formato binario común (**1 para positivo, 0 para negativo**).
* **Justificación del modelo recomendado:** **RoBERTuito** destaca como la opción más eficiente debido a su latencia de solo **~195.31 ms en CPU**, duplicando la velocidad de XLM-RoBERTa sin perder precisión, como lo demuestra su F1-Score perfecto de 1.0000. Al pesar apenas **435 MB**, reduce drásticamente el consumo de RAM y los costos de servidor, superando la pesada infraestructura de 1.1 GB de su competidor multilingüe. Además, su preentrenamiento nativo en español le permite entender a la perfección modismos, emojis y texto informal de redes sociales, con la gran ventaja de contar con una **licencia MIT** que habilita su explotación comercial sin las restricciones legales que limitan a modelos como BETO Sentiment.


## 3) Descripción y estructura del dataset de soporte

Para llevar a cabo una evaluación ágil, consistente y comparable de los modelos de procesamiento de lenguaje natural (NLP), se ha diseñado y estructurado un dataset de prueba simplificado pero altamente representativo en español.

* **Estructura del Dataset y Campos:** El conjunto de datos está constituido por registros estructurados con los siguientes campos principales:
  * `text` *(String)*: Frase, comentario o reseña de texto en español que expresa una opinión o experiencia de usuario.
  * `label` *(String)*: Etiqueta categórica de sentimiento asociado al texto.
* **Etiquetas y sentimientos:** Las etiquetas del dataset están definidas bajo un esquema binario de clasificación de la siguiente manera:
  * **`POS` (Positive):** Representa comentarios que contienen sentimientos de satisfacción, aprobación o recomendación explícita *(mapeado internamente a un valor numérico de 1)*.
  * **`NEG` (Negative):** Representa opiniones de insatisfacción, quejas, frustración o desaprobación *(mapeado internamente a un valor numérico de 0)*.

---

## 4) Consideraciones

* **Entorno de ejecución:** Se utilizó **Google Colab** con un entorno de ejecución de Python 3.x. No se utilizó GPU, la inferencia se procesó **solamente con CPU**.
* **Pasos de ejecución:** 
  1. **Verificación de Hardware:** Se comprueba la disponibilidad de aceleración gráfica por hardware.
  2. **Instalación e Importación:** Se descargan las librerías necesarias de Hugging Face (`transformers`, `datasets`) y las herramientas de evaluación para configurar el entorno.
  3. **Creación del Dataset de Prueba:** Se define una lista estructurada con 10 textos representativos en español junto con sus correspondientes etiquetas manuales (`POS` y `NEG`).
  4. **Configuración del Pipeline de Inferencia:** Se inicializa la tarea `text-classification` cargando el modelo específico (RoBERTuito) y enviándole cada uno de los textos de prueba secuencialmente.
  5. **Medición y Evaluación:** Se calcula el tiempo de respuesta por cada predicción hecha por el modelo y se comparan los resultados predichos con las etiquetas reales para generar las métricas de exactitud (`Accuracy`, `F1-Score` y `Latencia`).

### Conclusiones

* **RoBERT:** Presenta la menor latencia promedio por inferencia, lo que lo convierte en un excelente candidato para escenarios de producción de tiempo real y baja infraestructura, manteniendo una métrica perfecta (1.0000) en este dataset acotado.
* **BETO:** Es una opción robusta por ser entrenado nativamente en español de principio a fin, logrando un balance destacado, aunque con una latencia ligeramente mayor.
* **RoBERTa:** Muestra métricas perfectas para la tarea, pero al ser un modelo multilingüe de gran tamaño, su latencia tiende a ser más pesada y computacionalmente demandante.

> **Justificación Técnica:** Para entornos de prototipado rápido y producción donde la eficiencia es crítica, **RoBERTuito** destaca como la opción óptima por su rendimiento excepcional de latencia y adaptabilidad al lenguaje informal en español.

## 5) Repositorio

* **Archivos de soporte:**
  * Dataset local de pruebas (formato estructurado en Python / CVS).
  * Dataset de Hugging Face de referencia: [`paws-x`](https://docs.google.com/spreadsheets/d/10gvpfC2PjkMEej-xLwPfXUcV2H2RDkRf8oEHhZkPm28/edit?usp=sharing).
* **Referencias:**
