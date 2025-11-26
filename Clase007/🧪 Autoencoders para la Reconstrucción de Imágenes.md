

![p1](https://github.com/Grandote58/NeoDeepLab/blob/main/Clase007/assets/PRACTICA.png)

# **🧪 Autoencoders para la Reconstrucción de Imágenes**

**“Reconstrucción de Imágenes con Autoencoders Convolucionales en Deep Learning”**

## 🎯 Objetivo general

Diseñar, entrenar y evaluar un **autoencoder convolucional** para la reconstrucción de imágenes, utilizando un conjunto de datos abierto, analizando cada etapa del proceso (exploración, preprocesamiento, entrenamiento, validación y evaluación).

## 🎯 Metas de aprendizaje

Al finalizar esta práctica el estudiante será capaz de:

1. **Cargar y explorar** un dataset de imágenes de acceso abierto (MNIST).
2. **Preprocesar y normalizar** imágenes para usarlas en un autoencoder convolucional.
3. **Implementar en Keras/TensorFlow** la arquitectura de un autoencoder para reconstrucción de imágenes.
4. **Entrenar y validar** el modelo, analizando las curvas de pérdida.
5. **Evaluar la calidad de la reconstrucción** tanto cualitativamente (visualmente) como cuantitativamente (errores de reconstrucción).
6. **Interpretar el espacio latente** como representación comprimida de las imágenes.

## 🔧 Instrucciones iniciales para Google Colab

> En Colab, ve a:
>  **Entorno de ejecución → Cambiar tipo de entorno de ejecución → Acelerador por hardware: GPU**
>  y selecciona **GPU** para acelerar el entrenamiento.

# **🧱 Sección 0 — Importación de librerías y configuración global**

Crea una celda de código con lo siguiente:

```python
# ============================================
# SECCIÓN 0: IMPORTACIÓN DE LIBRERÍAS
# ============================================

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models

# Comprobar versión de TensorFlow
print("Versión de TensorFlow:", tf.__version__)

# Configurar un seed para reproducibilidad básica
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Comprobar si hay GPU disponible
device_name = tf.config.list_physical_devices('GPU')
print("Dispositivos GPU disponibles:", device_name)
```

# **📊 Sección 1 — Carga y exploración del conjunto de datos (datos abiertos)**

Usaremos **MNIST**, un dataset abierto de dígitos manuscritos (28x28, escala de grises).

### 🔹 1.1 Carga de datos

```python
# ============================================
# SECCIÓN 1: CARGA DEL DATASET (MNIST)
# ============================================

# MNIST viene incluido en tf.keras.datasets y es de acceso abierto
from tensorflow.keras.datasets import mnist

# Cargar datos: (x_train, y_train), (x_test, y_test)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Forma de x_train:", x_train.shape)
print("Forma de y_train:", y_train.shape)
print("Forma de x_test :", x_test.shape)
print("Forma de y_test :", y_test.shape)
```

### 🔹 1.2 Exploración básica de datos

```python
# Mostrar algunos ejemplos de imágenes y sus etiquetas originales
num_samples = 9
plt.figure(figsize=(6, 6))

for i in range(num_samples):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(f"Etiqueta: {y_train[i]}")
    plt.axis("off")

plt.suptitle("Ejemplos de imágenes del conjunto de entrenamiento (MNIST)", fontsize=14)
plt.tight_layout()
plt.show()
```

### 🔹 1.3 Distribución de etiquetas y valores de píxel

```python
# Distribución de clases (aunque no se usen para el autoencoder, sirven para explorar)
unique, counts = np.unique(y_train, return_counts=True)
print("Distribución de clases en y_train:")
for u, c in zip(unique, counts):
    print(f"Dígito {u}: {c} imágenes")

# Graficar histograma de intensidades de píxel
plt.figure(figsize=(6, 4))
plt.hist(x_train.reshape(-1), bins=50)
plt.title("Distribución de valores de píxel en x_train (0-255)")
plt.xlabel("Intensidad de píxel")
plt.ylabel("Frecuencia")
plt.show()
```

# **🧼 Sección 2 — Preprocesamiento de datos**

### Pasos clave:

- Normalizar intensidades a rango [0, 1].
- Añadir dimensión de canal: (28, 28, 1) para usar capas convolucionales.
- Dividir conjunto de validación desde el train (opcional pero recomendado).

```python
# ============================================
# SECCIÓN 2: PREPROCESAMIENTO
# ============================================

# Convertir a float32 y normalizar en el rango [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("Rango de valores después de normalizar:")
print("x_train min:", x_train.min(), "max:", x_train.max())
print("x_test  min:", x_test.min(), "max:", x_test.max())

# Añadir dimensión de canal (canal único para escala de grises)
x_train = np.expand_dims(x_train, axis=-1)  # (num_samples, 28, 28, 1)
x_test = np.expand_dims(x_test, axis=-1)

print("Nueva forma de x_train:", x_train.shape)
print("Nueva forma de x_test :", x_test.shape)

# Crear subconjunto de validación a partir de entrenamiento (por ejemplo, 10%)
val_fraction = 0.1
val_size = int(len(x_train) * val_fraction)

x_val = x_train[:val_size]
x_train_sub = x_train[val_size:]

print("Forma de x_train_sub:", x_train_sub.shape)
print("Forma de x_val      :", x_val.shape)
```

### 🔹 2.1 Visualización post-preprocesamiento

```python
# Verificar que las imágenes siguen correctas tras el preprocesamiento
plt.figure(figsize=(6, 3))

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(x_train_sub[i].squeeze(), cmap="gray")
    plt.title(f"Mín: {x_train_sub[i].min():.2f} Máx: {x_train_sub[i].max():.2f}")
    plt.axis("off")

plt.suptitle("Muestras tras normalización [0, 1]", fontsize=14)
plt.tight_layout()
plt.show()
```

# **🧠 Sección 3 — Definición de la arquitectura del Autoencoder Convolucional**

Diseñaremos un **autoencoder convolucional undercomplete**:

- **Encoder**:
  - Conv2D → ReLU → MaxPooling
  - Conv2D → ReLU → MaxPooling
- **Latent**:
  - Mapa de activación comprimido (por ejemplo 7x7x32).
- **Decoder**:
  - Conv2D → ReLU → UpSampling
  - Conv2D → ReLU → UpSampling
  - Conv2D (1 canal, activación sigmoide) → imagen reconstruida.

```python
# ============================================
# SECCIÓN 3: DEFINICIÓN DEL AUTOENCODER
# ============================================

input_shape = (28, 28, 1)

# Definición del modelo encoder
encoder_inputs = layers.Input(shape=input_shape, name="encoder_input")

# Bloque de convolución 1
x = layers.Conv2D(
    filters=32, 
    kernel_size=(3, 3), 
    activation="relu", 
    padding="same",
    name="enc_conv1"
)(encoder_inputs)
x = layers.MaxPooling2D((2, 2), padding="same", name="enc_pool1")(x)

# Bloque de convolución 2
x = layers.Conv2D(
    filters=64,
    kernel_size=(3, 3),
    activation="relu",
    padding="same",
    name="enc_conv2"
)(x)
encoded = layers.MaxPooling2D((2, 2), padding="same", name="enc_pool2")(x)

# encoded es el espacio latente en forma de mapa de características
print("Forma del espacio latente (encoded):", encoded.shape)

# Definición del decoder (simétrico aproximado)
x = layers.Conv2D(
    filters=64,
    kernel_size=(3, 3),
    activation="relu",
    padding="same",
    name="dec_conv1"
)(encoded)
x = layers.UpSampling2D((2, 2), name="dec_upsample1")(x)

x = layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation="relu",
    padding="same",
    name="dec_conv2"
)(x)
x = layers.UpSampling2D((2, 2), name="dec_upsample2")(x)

decoder_outputs = layers.Conv2D(
    filters=1,
    kernel_size=(3, 3),
    activation="sigmoid",  # para salida en [0, 1]
    padding="same",
    name="dec_conv_output"
)(x)

# Autoencoder completo: entrada -> salida reconstruida
autoencoder = models.Model(encoder_inputs, decoder_outputs, name="conv_autoencoder")

autoencoder.summary()
```

# **⚙️ Sección 4 — Compilación y entrenamiento del modelo**

Usaremos:

- Optimizador: **Adam**
- Pérdida: **binary_crossentropy** (adecuada para imágenes normalizadas en [0, 1])

```python
# ============================================
# SECCIÓN 4: COMPILACIÓN Y ENTRENAMIENTO
# ============================================

autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy"
)

# Entrenamiento
epochs = 15
batch_size = 256

history = autoencoder.fit(
    x_train_sub, x_train_sub,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(x_val, x_val)
)
```

### 🔹 4.1 Visualización de curvas de entrenamiento y validación

```
# Graficar la función de pérdida (loss) de entrenamiento y validación

plt.figure(figsize=(6, 4))
plt.plot(history.history["loss"], label="Pérdida de entrenamiento")
plt.plot(history.history["val_loss"], label="Pérdida de validación")
plt.title("Curvas de pérdida (autoencoder)")
plt.xlabel("Épocas")
plt.ylabel("Pérdida (binary_crossentropy)")
plt.legend()
plt.grid(True)
plt.show()
```

> **Interpretación:**
>
> - Curvas que disminuyen y se estabilizan sugieren un entrenamiento adecuado.
> - Si la pérdida de validación sube mientras la de entrenamiento baja, puede haber sobreajuste.

# **🧪 Sección 5 — Evaluación cualitativa: visualización de reconstrucciones**

Probaremos el modelo en el conjunto de prueba (**x_test**).

```python
# ============================================
# SECCIÓN 5: EVALUACIÓN CUALITATIVA
# ============================================

# Obtener reconstrucciones del conjunto de test
decoded_imgs = autoencoder.predict(x_test)

# Visualizar algunas imágenes originales vs reconstruidas
num_images = 10
plt.figure(figsize=(20, 4))

for i in range(num_images):
    # Imágenes originales
    ax = plt.subplot(2, num_images, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # Imágenes reconstruidas
    ax = plt.subplot(2, num_images, i + 1 + num_images)
    plt.imshow(decoded_imgs[i].squeeze(), cmap="gray")
    plt.title("Reconstruida")
    plt.axis("off")

plt.suptitle("Comparación: Original vs Reconstruida (Autoencoder)", fontsize=16)
plt.tight_layout()
plt.show()
```

> Aquí puedes comentar visualmente:
>
> - ¿Se conservan los contornos?
> - ¿Qué detalles se pierden?
> - ¿Hay signos de sobre-suavizado (blurring)?

# **📏 Sección 6 — Evaluación cuantitativa: error de reconstrucción**

Calcularemos el **error cuadrático medio (MSE)** por imagen y analizaremos su distribución.

```python
# ============================================
# SECCIÓN 6: EVALUACIÓN CUANTITATIVA
# ============================================

from sklearn.metrics import mean_squared_error

# Flatten para calcular el MSE por muestra (imagen)
x_test_flat = x_test.reshape((len(x_test), -1))
decoded_flat = decoded_imgs.reshape((len(decoded_imgs), -1))

# Calcular MSE por imagen
mse_per_image = np.mean(np.power(x_test_flat - decoded_flat, 2), axis=1)

print("Forma de mse_per_image:", mse_per_image.shape)
print("Ejemplos de MSE por imagen:", mse_per_image[:10])

# Estadísticas descriptivas
print("\nEstadísticas del error de reconstrucción (MSE):")
print("Mínimo:", np.min(mse_per_image))
print("Máximo:", np.max(mse_per_image))
print("Media :", np.mean(mse_per_image))
print("Mediana:", np.median(mse_per_image))
```

### 🔹 6.1 Histograma de errores de reconstrucción

```python
plt.figure(figsize=(6, 4))
plt.hist(mse_per_image, bins=50)
plt.title("Distribución del error de reconstrucción (MSE) en el conjunto de test")
plt.xlabel("MSE por imagen")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.show()
```

> Este histograma te ayuda a ver:
>
> - ¿La mayoría de las imágenes se reconstruyen con error bajo?
> - ¿Hay colas largas que podrían interpretarse como “anomalías” en una aplicación de detección de anomalías?

# **🔍 Sección 7 — Análisis del espacio latente (opcional avanzado)**

Podemos extraer solo el **encoder** para obtener el espacio latente de las imágenes.

```python
# ============================================
# SECCIÓN 7: ANÁLISIS DEL ESPACIO LATENTE (OPCIONAL)
# ============================================

# Definimos un modelo encoder que termina en 'encoded'
encoder = models.Model(inputs=encoder_inputs, outputs=encoded, name="encoder_model")
encoder.summary()

# Obtenemos representaciones latentes de algunas imágenes de test
latent_representations = encoder.predict(x_test[:1000])  # por ejemplo, 1000 imágenes
print("Forma de latent_representations:", latent_representations.shape)

# Convertir a 2D para visualización (promediando canales y aplanando algo)
latent_flat = latent_representations.reshape((latent_representations.shape[0], -1))
print("Forma de latent_flat:", latent_flat.shape)
```

Puedes posteriormente usar **PCA o t-SNE** para visualizar, pero eso ya sería extensión.

## ⚠️ Advertencias técnicas y buenas prácticas

1. ##### **Normalización**:

    Asegúrate de que las imágenes estén normalizadas entre 0 y 1 cuando uses `sigmoid` como activación de salida y `binary_crossentropy` como pérdida.

2. ##### **Tamaño del espacio latente**:

   - Muy pequeño → pérdida de detalles, mala reconstrucción.
   - Muy grande → poca compresión, riesgo de memorizar.

3. ##### **Capacidad del modelo**:

    Ajusta el número de filtros y la profundidad del encoder/decoder según el tamaño del dataset y recursos computacionales.

4. ##### **Curvas de entrenamiento**:

    Monitoriza siempre `loss` y `val_loss` para detectar **sobreajuste**.

5. ##### **Uso de GPU**:

    Esta práctica es mucho más eficiente con GPU. Si no está activa, el tiempo de entrenamiento será mayor.

## ✅ Resumen de la práctica

En esta práctica has:

1. Trabajado con un **dataset abierto** (MNIST).
2. Preprocesado y visualizado imágenes, analizando su distribución.
3. Definido un **autoencoder convolucional** en Keras/TensorFlow.
4. Entrenado, validado y graficado curvas de pérdida.
5. Evaluado cualitativamente las reconstrucciones (imágenes).
6. Evaluado cuantitativamente el rendimiento mediante el **MSE**.
7. Extraído representaciones latentes para análisis futuro.


![p2](https://github.com/Grandote58/NeoDeepLab/blob/main/Clase007/assets/pie.png)
