![PRACTICA](D:\001_Uniminuto2025\005DeepLearning\Clase008\Recursos\assets\PRACTICA.png)

# **🧪 Práctica: Detección de Fraude Bancario con Autoencoder en Keras/TensorFlow 2**

## 🎯 Objetivo general

Diseñar, entrenar y evaluar un **autoencoder denso** para la detección de fraudes bancarios, utilizando un dataset financiero abierto y aplicando un enfoque de **detección de anomalías** basado en error de reconstrucción.

## 🎯 Metas de aprendizaje

Al finalizar la práctica, el estudiante será capaz de:

1. Cargar y explorar un conjunto de datos abierto de transacciones bancarias (fraude con tarjeta de crédito).
2. Preprocesar variables numéricas para usarlas en un autoencoder (escalado, separación de clases, balanceo conceptual).
3. Implementar un **autoencoder en Keras/TensorFlow 2.x** para modelar el comportamiento normal de las transacciones.
4. Entrenar y validar el modelo, analizando curvas de pérdida y distribución del error de reconstrucción.
5. Establecer un **umbral de anomalía** y evaluar el sistema con métricas como matriz de confusión, precision, recall y F1-score.
6. Interpretar los resultados y discutir ventajas, limitaciones y riesgos del enfoque.



> ✅ **Instrucción clave para Colab**
>  Antes de empezar: ve a **Entorno de ejecución → Cambiar tipo de entorno de ejecución → Acelerador por hardware: GPU (opcional)** para acelerar el entrenamiento.

# **🧱 Sección 0 — Configuración inicial y librerías**

Crea una celda de código en Colab con lo siguiente:

```python
# ============================================
# SECCIÓN 0: IMPORTACIÓN DE LIBRERÍAS Y SETUP
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, precision_recall_curve)

import tensorflow as tf
from tensorflow.keras import layers, models

# Mostrar versión de TensorFlow
print("Versión de TensorFlow:", tf.__version__)

# Semillas para reproducibilidad
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

# **📥 Sección 1 — Carga y visualización del dataset (datos abiertos)**

Usaremos el dataset público **Credit Card Fraud Detection** (Europa, 2013). Muchas copias están disponibles de forma abierta. En Colab podemos descargarlo desde un repositorio público.

> 🔎 Opción típica: un repositorio público de GitHub que contenga `creditcard.csv`.
>  (Al ejecutar en Colab, asegúrate de que la URL esté accesible. Te dejo un ejemplo genérico con una URL de GitHub; si cambiara en el futuro, solo debes sustituirla por una URL pública de `creditcard.csv`.)

```python
# ============================================
# SECCIÓN 1: DESCARGA Y CARGA DEL DATASET
# ============================================

# EJEMPLO: descarga desde un repositorio público (ajusta la URL si usas otra fuente)
!wget -q https://raw.githubusercontent.com/omdomg/creditcard-fraud-detection/master/creditcard.csv -O creditcard.csv

# Cargar el CSV
data = pd.read_csv("creditcard.csv")

# Mostrar dimensiones del dataset
print("Dimensiones del dataset:", data.shape)

# Primeras filas
display(data.head())

# Información de tipos de datos
print("\nInformación del dataset:")
print(data.info())
```

# **📊 Sección 2 — Exploración inicial (EDA básica)**

### 🔹 2.1 Distribución de la variable objetivo (fraude vs no fraude)

```python
# ============================================
# SECCIÓN 2: EXPLORACIÓN DE DATOS
# ============================================

# Ver distribución de la variable 'Class' (0 = no fraude, 1 = fraude)
class_counts = data['Class'].value_counts()
print("Distribución de clases:\n", class_counts)

# Porcentajes
fraud_percentage = class_counts[1] / class_counts.sum() * 100
print(f"\nPorcentaje de fraudes: {fraud_percentage:.4f}%")

# Gráfico de barras
plt.figure(figsize=(6,4))
class_counts.plot(kind='bar')
plt.title("Distribución de clases (0 = Normal, 1 = Fraude)")
plt.xlabel("Clase")
plt.ylabel("Número de transacciones")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()
```

### 🔹 2.2 Estadísticos descriptivos y distribución de algunos campos

```python
# Descripción estadística de las variables numéricas
display(data.describe().T.head(10))

# Histograma del monto de transacción (Amount)
plt.figure(figsize=(6,4))
plt.hist(data['Amount'], bins=50)
plt.title("Distribución del monto de las transacciones")
plt.xlabel("Amount")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.show()

# Histograma de la variable 'Time'
plt.figure(figsize=(6,4))
plt.hist(data['Time'], bins=50)
plt.title("Distribución de la variable 'Time'")
plt.xlabel("Time (segundos desde primera transacción)")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.show()
```

# **🧼 Sección 3 — Preprocesamiento de datos**

### Decisiones de diseño:

- Las variables `V1` a `V28` ya son componentes PCA anonimizados.
- Normalizaremos **Amount** y **Time**, y luego escalaremos todo el vector de características.
- Usaremos solo variables numéricas para el autoencoder.

### 🔹 3.1 Separación de características y etiqueta

```python
# ============================================
# SECCIÓN 3: PREPROCESAMIENTO
# ============================================

# Separar características (X) y etiqueta (y)
X = data.drop('Class', axis=1)
y = data['Class']

print("Shape de X:", X.shape)
print("Shape de y:", y.shape)
```

### 🔹 3.2 Escalado de variables (StandardScaler)

```python
# Escalar todas las características con StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("X_scaled shape:", X_scaled.shape)

# Convertir a DataFrame para inspección
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

display(X_scaled_df.describe().T.head(10))
```

# **🧪 Sección 4 — División normal vs fraude, y partición train/test**

El autoencoder se entrenará **solo con transacciones normales (Class=0)**.

```python
# ============================================
# SECCIÓN 4: DIVISIÓN NORMAL / FRAUDE Y TRAIN/TEST
# ============================================

# Índices de normales y fraudes
normal_mask = (y == 0)
fraud_mask = (y == 1)

X_normal = X_scaled[normal_mask]
X_fraud = X_scaled[fraud_mask]

print("Transacciones normales:", X_normal.shape[0])
print("Transacciones fraude  :", X_fraud.shape[0])

# Dividimos normales en train y test
from sklearn.model_selection import train_test_split

X_train_normal, X_test_normal = train_test_split(
    X_normal, test_size=0.2, random_state=SEED
)

# Para evaluación, combinamos normales de test + todas las fraudes
X_test_combined = np.vstack([X_test_normal, X_fraud])
y_test_combined = np.hstack([np.zeros(len(X_test_normal)), np.ones(len(X_fraud))])

print("\nShape X_train_normal:", X_train_normal.shape)
print("Shape X_test_normal :", X_test_normal.shape)
print("Shape X_test_combined:", X_test_combined.shape)
print("Shape y_test_combined:", y_test_combined.shape)
```

### 🔹 4.1 Visualización simple del desbalance en test combinado

```python
unique_test, counts_test = np.unique(y_test_combined, return_counts=True)
print("\nDistribución en el conjunto de test combinado:")
for u, c in zip(unique_test, counts_test):
    print(f"Clase {u}: {c} transacciones")

plt.figure(figsize=(4,4))
plt.bar(['Normal', 'Fraude'], counts_test)
plt.title("Distribución en test combinado")
plt.ylabel("Número de transacciones")
plt.grid(axis='y')
plt.show()
```

# **🧠 Sección 5 — Definición del Autoencoder en Keras/TensorFlow 2**

Usaremos un modelo fully-connected (denso) para datos tabulares.

### 🔹 5.1 Arquitectura del autoencoder

```python
# ============================================
# SECCIÓN 5: DEFINICIÓN DEL AUTOENCODER
# ============================================

input_dim = X_train_normal.shape[1]
encoding_dim = 16  # dimensión del espacio latente (bottleneck)

# Definición con API funcional (más clara)
input_layer = layers.Input(shape=(input_dim,), name="input")

# Encoder
x = layers.Dense(64, activation='relu', name="enc_dense1")(input_layer)
x = layers.Dense(32, activation='relu', name="enc_dense2")(x)
latent = layers.Dense(encoding_dim, activation='relu', name="latent")(x)

# Decoder
x = layers.Dense(32, activation='relu', name="dec_dense1")(latent)
x = layers.Dense(64, activation='relu', name="dec_dense2")(x)
output_layer = layers.Dense(input_dim, activation='linear', name="output")(x)

autoencoder = models.Model(inputs=input_layer, outputs=output_layer, name="autoencoder_fraude")

autoencoder.summary()
```

# **⚙️ Sección 6 — Compilación y entrenamiento del Autoencoder**

Entrenaremos el modelo como una regresión: entrada ≈ salida, ambas son `X_train_normal`.

```python
# ============================================
# SECCIÓN 6: COMPILACIÓN Y ENTRENAMIENTO
# ============================================

autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='mse'   # pérdida de reconstrucción
)

# Entrenamiento
EPOCHS = 30
BATCH_SIZE = 256

history = autoencoder.fit(
    X_train_normal, X_train_normal,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split=0.1,
    verbose=1
)
```

### 🔹 6.1 Gráficas de pérdida de entrenamiento y validación

```python
plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.title("Curva de pérdida del Autoencoder")
plt.xlabel("Épocas")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)
plt.show()
```

> 💡 **Interpretación:**
>
> - Pérdida decreciente y estabilizada = entrenamiento razonable.
> - Si `val_loss` sube mientras `loss` baja → posible sobreajuste.

------

# **🧾 Sección 7 — Cálculo del error de reconstrucción**

Calculamos el error MSE de reconstrucción en **todo el conjunto de test combinado** (normales + fraudes).

```python
# ============================================
# SECCIÓN 7: ERROR DE RECONSTRUCCIÓN EN TEST
# ============================================

# Obtener reconstrucciones
reconstructions = autoencoder.predict(X_test_combined)

# Error MSE por transacción
mse = np.mean(np.power(X_test_combined - reconstructions, 2), axis=1)

print("Shape mse:", mse.shape)
print("Primeros 10 errores MSE:", mse[:10])
```

### 🔹 7.1 Distribución del error de reconstrucción

```python
plt.figure(figsize=(6,4))
plt.hist(mse, bins=50)
plt.title("Distribución del error de reconstrucción (MSE) - Test combinado")
plt.xlabel("MSE")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.show()
```

### 🔹 7.2 Comparar errores entre normales y fraudes

```python
# Separar errores para normales y fraudes
mse_normal = mse[y_test_combined == 0]
mse_fraud = mse[y_test_combined == 1]

print("MSE normal - media:", np.mean(mse_normal), "mediana:", np.median(mse_normal))
print("MSE fraude - media:", np.mean(mse_fraud), "mediana:", np.median(mse_fraud))

plt.figure(figsize=(6,4))
plt.hist(mse_normal, bins=50, alpha=0.7, label='Normal')
plt.hist(mse_fraud,  bins=50, alpha=0.7, label='Fraude')
plt.title("Distribución MSE: normales vs fraudes")
plt.xlabel("MSE")
plt.ylabel("Frecuencia")
plt.legend()
plt.grid(True)
plt.show()
```

> 🔎 Esperado: los fraudes tienden a tener errores más altos que las transacciones normales.

# **🔐 Sección 8 — Selección de umbral de anomalía**

Elegiremos un umbral basado en el **percentil** de los errores sobre transacciones normales de test.

```python
# ============================================
# SECCIÓN 8: UMBRAL DE ANOMALÍA
# ============================================

# Umbral basado solo en errores de las normales
threshold = np.percentile(mse_normal, 95)  # p.ej. percentil 95

print("Umbral seleccionado (percentil 95 de normales):", threshold)
```

Podrías probar otros percentiles (97, 99) para ajustar la sensibilidad.

# **🧮 Sección 9 — Clasificación final y métricas**

### 🔹 9.1 Clasificación según el umbral

```python
# ============================================
# SECCIÓN 9: CLASIFICACIÓN Y MÉTRICAS
# ============================================

# Predicción: 1 = fraude si error >= umbral
y_pred = (mse >= threshold).astype(int)

print("Primeras 20 predicciones:", y_pred[:20])
print("Primeros 20 valores reales:", y_test_combined[:20])
```

### 🔹 9.2 Matriz de confusión y métricas clásicas

```python
cm = confusion_matrix(y_test_combined, y_pred)
print("Matriz de confusión:\n", cm)

tn, fp, fn, tp = cm.ravel()
print(f"\nTN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

print("\nReporte de clasificación:")
print(classification_report(y_test_combined, y_pred, digits=4))

precision = precision_score(y_test_combined, y_pred)
recall = recall_score(y_test_combined, y_pred)
f1 = f1_score(y_test_combined, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
```

### 🔹 9.3 ROC-AUC y curva ROC

```python
# ROC-AUC usando los errores como score (mayor error = más probable fraude)
roc_auc = roc_auc_score(y_test_combined, mse)
print("ROC-AUC (usando MSE como score):", roc_auc)

fpr, tpr, _ = roc_curve(y_test_combined, mse)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0,1], [0,1], 'k--', label="Random")
plt.title("Curva ROC - Detección de fraude con Autoencoder")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()
```

### 🔹 9.4 Curva Precision–Recall

```python
precision_vals, recall_vals, _ = precision_recall_curve(y_test_combined, mse)

plt.figure(figsize=(6,4))
plt.plot(recall_vals, precision_vals)
plt.title("Curva Precision-Recall - Detección de fraude")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.show()
```

![pie](D:\001_Uniminuto2025\005DeepLearning\Clase008\Recursos\assets\pie.png)

