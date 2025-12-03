![PRACTICA](https://github.com/Grandote58/NeoDeepLab/blob/main/Clase008/assets/PRACTICA.png)


# **🧪 Práctica: Denoising Autoencoder para Detección de Fraude Bancario**

## 🎯 Objetivo general

Implementar, entrenar y evaluar un **Denoising Autoencoder** aplicado a transacciones bancarias (dataset de fraude con tarjetas de crédito), con el fin de:

1. **Modelar el comportamiento normal** de las transacciones bajo presencia de ruido.
2. **Distinguir ruido aleatorio** de **anomalías estructuradas** (fraude) a partir del **error de reconstrucción**.

## 🎯 Metas de aprendizaje

Al finalizar la práctica, el estudiante será capaz de:

1. Cargar y explorar un **dataset financiero abierto** de fraude con tarjetas de crédito.
2. Identificar y **simular ruido** en las características de las transacciones.
3. Implementar un **Denoising Autoencoder** en Keras/TensorFlow 2.x que aprenda a “limpiar” ruido de transacciones normales.
4. Analizar el **error de reconstrucción** como indicador de anomalía y derivar de él una regla de decisión.
5. Evaluar el desempeño de la técnica mediante métricas estándar (confusion matrix, precision, recall, F1, ROC-AUC, PR curve).



> 🔧 **Antes de empezar en Colab**
>  Menú: **Entorno de ejecución → Cambiar tipo de entorno de ejecución → Acelerador por hardware: GPU (opcional pero recomendado)**.

# **🧱 SECCIÓN 0 — Librerías y configuración inicial**

```python
# ============================================
# SECCIÓN 0: IMPORTACIÓN DE LIBRERÍAS Y SETUP
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, precision_recall_curve)

import tensorflow as tf
from tensorflow.keras import layers, models

# Configurar estilo de gráficas
plt.style.use("seaborn-v0_8")

# Mostrar versión de TensorFlow
print("Versión de TensorFlow:", tf.__version__)

# Semillas para reproducibilidad
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

# **📥 SECCIÓN 1 — Carga del dataset abierto y vista general**

Usaremos el dataset público **Credit Card Fraud** alojado por TensorFlow (misma estructura que el clásico de Kaggle).

```python
# ============================================
# SECCIÓN 1: DESCARGA Y CARGA DEL DATASET
# ============================================

# Descarga del dataset (datos abiertos)
!wget -q https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv -O creditcard.csv

# Cargar CSV en un DataFrame
data = pd.read_csv("creditcard.csv")

print("Dimensiones del dataset:", data.shape)
print("\nPrimeras filas:")
display(data.head())

print("\nInformación del dataset:")
print(data.info())
```

# **📊 SECCIÓN 2 — Exploración de datos (EDA) y ruido implícito**

### 🔹 2.1 Distribución de la variable Clase (0 vs 1)

```python
# ============================================
# SECCIÓN 2: EXPLORACIÓN DE DATOS
# ============================================

class_counts = data["Class"].value_counts()
print("Distribución de clases:\n", class_counts)

fraud_pct = class_counts[1] / class_counts.sum() * 100
print(f"\nPorcentaje de fraudes: {fraud_pct:.4f}%")

plt.figure(figsize=(6,4))
class_counts.plot(kind="bar", color=["tab:blue", "tab:red"])
plt.title("Distribución de clases (0 = Normal, 1 = Fraude)")
plt.xlabel("Clase")
plt.ylabel("Número de transacciones")
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.show()
```

> 💡 Reflexión: el **fraude es muy raro** → el ruido/varianza de comportamiento normal es mucho mayor en volumen que los fraudes.

### 🔹 2.2 Estadísticos de variables clave: Time y Amount

```python
print("\nDescripción estadística de 'Time' y 'Amount':")
display(data[["Time", "Amount"]].describe().T)

# Histograma de Amount
plt.figure(figsize=(6,4))
plt.hist(data["Amount"], bins=50)
plt.title("Distribución del monto (Amount)")
plt.xlabel("Amount")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.show()

# Histograma de Time
plt.figure(figsize=(6,4))
plt.hist(data["Time"], bins=50)
plt.title("Distribución de 'Time'")
plt.xlabel("Time (segundos desde la primera transacción)")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.show()
```

> 💬 Comentario técnico: la variabilidad (ruido) natural en montos y tiempos es alta. El autoencoder debe aprender qué combinaciones de variables son “normales” a pesar de esa variación.

# **🧼 SECCIÓN 3 — Preprocesamiento y definición de ruido**

### Idea clave de la técnica

- Trabajaremos con las variables numéricas (`Time`, `V1`–`V28`, `Amount`).
- **Entrenaremos un Denoising Autoencoder solo con transacciones normales**:
  - **Entrada**: transacción **normal + ruido gaussiano sintético**.
  - **Salida (target)**: transacción normal original (sin ruido).
- De este modo, el modelo aprende a **filtrar ruido aleatorio** y reconstruir el “patrón limpio” de comportamiento normal.
- Cuando vea transacciones fraudulentas o muy extrañas, las tratará como “ruido estructurado” difícil de limpiar → **error de reconstrucción alto**.

### 🔹 3.1 Separar X e y

```python
# ============================================
# SECCIÓN 3: PREPROCESAMIENTO
# ============================================

X = data.drop("Class", axis=1)
y = data["Class"]

print("Shape X:", X.shape)
print("Shape y:", y.shape)
```

### 🔹 3.2 Escalado de características

Usamos `StandardScaler` para que el autoencoder trabaje con distribuciones centradas y comparables.

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Shape X_scaled:", X_scaled.shape)

X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
print("\nDescripción estadística tras escalar:")
display(X_scaled_df.describe().T.head(10))
```

### 🔹 3.3 Separación Normal vs Fraude

```python
normal_mask = (y == 0)
fraud_mask = (y == 1)

X_normal = X_scaled[normal_mask]
X_fraud  = X_scaled[fraud_mask]

print("Transacciones normales:", X_normal.shape[0])
print("Transacciones fraude  :", X_fraud.shape[0])
```

# **🌫️ SECCIÓN 4 — Generación explícita de ruido (Gaussiano) y justificación**

### Concepto de ruido en esta práctica

- **Ruido aleatorio (sintético)**: pequeñas perturbaciones gaussianas que afectan las variables pero no cambian la “naturaleza” de la transacción (sigue siendo normal).
- **Fraude**: no lo generamos artificialmente; ya está en el dataset. Es una **anomalía estructurada**, no solo ruido pequeño.

### 🔹 4.1 Función para añadir ruido gaussiano a transacciones normales

```python
# ============================================
# SECCIÓN 4: GENERACIÓN DE RUIDO
# ============================================

def add_gaussian_noise(X, mean=0.0, std=0.05):
    """
    Añade ruido gaussiano a una matriz X.
    
    X: np.array de forma (n_muestras, n_features)
    mean: media del ruido
    std: desviación estándar del ruido
    """
    noise = np.random.normal(loc=mean, scale=std, size=X.shape)
    X_noisy = X + noise
    return X_noisy

# Aplicar ruido sobre transacciones normales
X_normal_noisy = add_gaussian_noise(X_normal, mean=0.0, std=0.1)

print("Shape X_normal:", X_normal.shape)
print("Shape X_normal_noisy:", X_normal_noisy.shape)
```

### 🔹 4.2 Comparar distribuciones con y sin ruido (una feature de ejemplo)

```python
# Elegimos arbitrariamente una columna, por ejemplo 'Amount'
col_name = "Amount"
col_idx = list(X.columns).index(col_name)

plt.figure(figsize=(8,4))
plt.hist(X_normal[:, col_idx], bins=50, alpha=0.7, label="Original")
plt.hist(X_normal_noisy[:, col_idx], bins=50, alpha=0.7, label="Con ruido")
plt.title(f"Distribución de {col_name}: original vs con ruido gaussiano")
plt.xlabel("Valor escalado")
plt.ylabel("Frecuencia")
plt.legend()
plt.grid(True)
plt.show()
```

> 🧠 **Lectura técnica:** el ruido gaussiano simula pequeñas perturbaciones naturales, errores de medición o variaciones menores. El autoencoder aprenderá a ignorar ese ruido y reconstruir la forma base de la transacción.

# **🔀 SECCIÓN 5 — Train/Test para el Denoising Autoencoder**

Entrenaremos el modelo **solo con normales**:

- **Entrada de entrenamiento**: `X_normal_noisy`
- **Target de entrenamiento**: `X_normal`

Para evaluación de fraude:

- Usaremos un **test combinado** con normales y fraudes **sin ruido sintético** (el “ruido” relevante ahora será el propio comportamiento anómalo del fraude).

```python
# ============================================
# SECCIÓN 5: SPLIT TRAIN / TEST PARA EL DENOISING
# ============================================

# Dividir las transacciones normales (originales y con ruido) en train/test
X_train_clean, X_val_clean, X_train_noisy, X_val_noisy = train_test_split(
    X_normal, X_normal_noisy, test_size=0.2, random_state=SEED
)

print("X_train_clean:", X_train_clean.shape)
print("X_val_clean  :", X_val_clean.shape)
print("X_train_noisy:", X_train_noisy.shape)
print("X_val_noisy  :", X_val_noisy.shape)

# Conjunto de test final para detección de fraude:
# normales (limpios) + fraudes (limpios)
X_test_combined = np.vstack([X_val_clean, X_fraud])
y_test_combined = np.hstack([np.zeros(len(X_val_clean)), np.ones(len(X_fraud))])

print("\nX_test_combined:", X_test_combined.shape)
print("y_test_combined:", y_test_combined.shape)
```

# **🧠 SECCIÓN 6 — Definición del Denoising Autoencoder**

Arquitectura simple pero suficiente:

- Entrada: vector de características con ruido (noisy).
- Encoder: Dense 64 → 32 → bottleneck 16.
- Decoder: Dense 32 → 64 → salida lineal (reconstrucción limpia).

```python
# ============================================
# SECCIÓN 6: MODELO DENOISING AUTOENCODER
# ============================================

input_dim = X_train_noisy.shape[1]
encoding_dim = 16

input_layer = layers.Input(shape=(input_dim,), name="input_noisy")

# Encoder
x = layers.Dense(64, activation='relu', name="enc_dense1")(input_layer)
x = layers.Dense(32, activation='relu', name="enc_dense2")(x)
latent = layers.Dense(encoding_dim, activation='relu', name="latent")(x)

# Decoder
x = layers.Dense(32, activation='relu', name="dec_dense1")(latent)
x = layers.Dense(64, activation='relu', name="dec_dense2")(x)
output_layer = layers.Dense(input_dim, activation='linear', name="output_clean")(x)

denoising_autoencoder = models.Model(inputs=input_layer,
                                     outputs=output_layer,
                                     name="denoising_autoencoder_fraude")

denoising_autoencoder.summary()
```

> 🎯 **Técnica subrayada:**
>  Estamos entrenando un modelo que “aprende a quitar ruido” de transacciones normales.
>  El fraude no es visto en el entrenamiento → se comporta como “ruido estructural” en test.

# **⚙️ SECCIÓN 7 — Compilación y entrenamiento (con validación)**

```python
# ============================================
# SECCIÓN 7: COMPILACIÓN Y ENTRENAMIENTO
# ============================================

denoising_autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse"
)

EPOCHS = 30
BATCH_SIZE = 256

history = denoising_autoencoder.fit(
    X_train_noisy, X_train_clean,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(X_val_noisy, X_val_clean),
    verbose=1
)
```

### 🔹 7.1 Curvas de pérdida (train vs val)

```python
plt.figure(figsize=(6,4))
plt.plot(history.history["loss"], label="Pérdida entrenamiento")
plt.plot(history.history["val_loss"], label="Pérdida validación")
plt.title("Curva de pérdida - Denoising Autoencoder")
plt.xlabel("Épocas")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)
plt.show()
```

> 💡 *Si hay divergencia fuerte entre loss y val_loss, probablemente haya sobreajuste o ruido excesivo.*

# **🧾 SECCIÓN 8 — Evaluación como detector de anomalías (fraudes)**

Ahora usamos el modelo entrenado para reconstruir **X_test_combined** (normales + fraudes, sin ruido sintético).

### 🔹 8.1 Reconstrucción y error MSE

```python
# ============================================
# SECCIÓN 8: RECONSTRUCCIÓN Y ERROR EN TEST
# ============================================

# Reconstrucción con el autoencoder (entrada: datos limpios)
reconstructions_test = denoising_autoencoder.predict(X_test_combined)

# Error MSE por muestra
mse_test = np.mean(np.power(X_test_combined - reconstructions_test, 2), axis=1)

print("Shape mse_test:", mse_test.shape)
print("Primeros 10 MSE:", mse_test[:10])
```

### 🔹 8.2 Distribución global de errores

```python
plt.figure(figsize=(6,4))
plt.hist(mse_test, bins=50)
plt.title("Distribución del error de reconstrucción - Test combinado")
plt.xlabel("MSE")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.show()
```

### 🔹 8.3 Separar error en normales vs fraudes

```python
mse_normal_test = mse_test[y_test_combined == 0]
mse_fraud_test  = mse_test[y_test_combined == 1]

print("MSE Normal - media:", np.mean(mse_normal_test), "mediana:", np.median(mse_normal_test))
print("MSE Fraude - media:", np.mean(mse_fraud_test), "mediana:", np.median(mse_fraud_test))

plt.figure(figsize=(6,4))
plt.hist(mse_normal_test, bins=50, alpha=0.7, label="Normal")
plt.hist(mse_fraud_test,  bins=50, alpha=0.7, label="Fraude")
plt.title("MSE: Normales vs Fraudes (Denoising AE)")
plt.xlabel("MSE")
plt.ylabel("Frecuencia")
plt.legend()
plt.grid(True)
plt.show()
```

> 🧠 **Lectura técnica de la técnica:**
>
> - Para el modelo, el fraude se comporta como un “ruido” que no sabe limpiar.
> - El Denoising AE sabe eliminar ruido gaussiano pequeño, pero el fraude altera la estructura multimodal de los datos ⇒ mayor error de reconstrucción.

# **🔐 SECCIÓN 9 — Selección del umbral (técnica de detección)**

Aquí puntualizamos cómo se genera la detección.

### Técnica de decisión (muy importante)

- Definimos un **umbral de error** `T`.
- Si `MSE(transacción) ≥ T` → la transacción se clasifica como **anómala/fraude**.
- Si `MSE(transacción) < T` → la transacción se clasifica como **normal**.

Elegiremos T como un percentil alto del MSE de las transacciones normales de test (`mse_normal_test`).

```python
# ============================================
# SECCIÓN 9: UMBRAL PARA DETECCIÓN DE FRAUDE
# ============================================

threshold = np.percentile(mse_normal_test, 97)  # por ejemplo, percentil 97
print("Umbral de anomalía (percentil 97 de normales):", threshold)
```

## 🧮 SECCIÓN 10 — Clasificación final y métricas

### 🔹 10.1 Etiquetas predichas

```python
# ============================================
# SECCIÓN 10: CLASIFICACIÓN Y MÉTRICAS
# ============================================

y_pred = (mse_test >= threshold).astype(int)

print("Primeras 20 predicciones:", y_pred[:20])
print("Primeros 20 valores reales:", y_test_combined[:20])
```

### 🔹 10.2 Matriz de confusión y métricas clave

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

### 🔹 10.3 ROC-AUC y curva ROC (usando el MSE como score)

```python
roc_auc = roc_auc_score(y_test_combined, mse_test)
print("ROC-AUC (Denoising AE, MSE como score):", roc_auc)

fpr, tpr, _ = roc_curve(y_test_combined, mse_test)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
plt.plot([0,1], [0,1], "k--", label="Random")
plt.title("Curva ROC - Denoising Autoencoder para fraude")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()
```

### 🔹 10.4 Curva Precision–Recall

```python
prec_vals, rec_vals, _ = precision_recall_curve(y_test_combined, mse_test)

plt.figure(figsize=(6,4))
plt.plot(rec_vals, prec_vals)
plt.title("Curva Precision-Recall - Denoising AE")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.show()
```


![pie](https://github.com/Grandote58/NeoDeepLab/blob/main/Clase008/assets/pie.png)

