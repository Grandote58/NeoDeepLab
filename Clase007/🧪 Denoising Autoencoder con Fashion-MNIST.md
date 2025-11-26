![PRACTICA](D:\001_Uniminuto2025\IMAGENES\PRACTICA.png)

# **🧪 Denoising Autoencoder con Fashion-MNIST**

**“Eliminación de ruido en imágenes de Fashion-MNIST usando Denoising Autoencoders”**

# **🎯 Objetivo general**

Implementar, entrenar y evaluar un **Denoising Autoencoder convolucional** capaz de reconstruir imágenes limpias de Fashion-MNIST a partir de versiones con ruido, analizando detalladamente cada etapa del proceso: exploración de datos, generación de ruido, preprocesamiento, entrenamiento, validación y evaluación cuantitativa y cualitativa.

# **🎯 Metas de aprendizaje**

Al finalizar la práctica, el estudiante será capaz de:

1. Cargar y explorar el dataset abierto **Fashion-MNIST**.
2. Generar versiones ruidosas de las imágenes mediante **ruido gaussiano** controlado.
3. Preprocesar las imágenes y preparar pares `(ruidosa → limpia)` para entrenar un **Denoising Autoencoder**.
4. Definir, entrenar y validar un modelo de autoencoder convolucional en Keras/TensorFlow.
5. Visualizar comparativamente imágenes **originales**, **ruidosas** y **reconstruidas**.
6. Calcular métricas de **error de reconstrucción** y analizar la distribución del error.
7. Identificar consideraciones técnicas clave en el diseño y entrenamiento de autoencoders para reconstrucción de imágenes.

------

# **🧱 SECCIÓN 0 – Importación de librerías y configuración**

```python
# ============================================
# SECCIÓN 0: IMPORTACIÓN DE LIBRERÍAS
# ============================================

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models

# Versión de TensorFlow
print("Versión de TensorFlow:", tf.__version__)

# Fijar semillas para reproducibilidad básica
SEED = 123
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Comprobar si hay GPU disponible
gpus = tf.config.list_physical_devices('GPU')
print("Dispositivos GPU disponibles:", gpus)
```

# **📊 SECCIÓN 1 – Carga y exploración del dataset Fashion-MNIST**

### 🔹 1.1 Cargar datos abiertos Fashion-MNIST

```python
# ============================================
# SECCIÓN 1: CARGA DEL DATASET FASHION-MNIST
# ============================================

from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print("Forma de x_train:", x_train.shape)
print("Forma de y_train:", y_train.shape)
print("Forma de x_test :", x_test.shape)
print("Forma de y_test :", y_test.shape)
```

### 🔹 1.2 Mapeo de clases (informativo)

```python
# Mapeo de etiquetas a nombres de clases (solo informativo, el autoencoder no usa y)
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Mostrar algunos ejemplos con sus etiquetas
num_samples = 9
plt.figure(figsize=(6, 6))
for i in range(num_samples):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(class_names[y_train[i]])
    plt.axis("off")

plt.suptitle("Ejemplos del conjunto Fashion-MNIST (train)", fontsize=14)
plt.tight_layout()
plt.show()
```

### 🔹 1.3 Distribución de clases y valores de píxel

```python
# Distribución de etiquetas
unique, counts = np.unique(y_train, return_counts=True)
print("Distribución de clases en y_train:")
for u, c in zip(unique, counts):
    print(f"{u} ({class_names[u]}): {c} imágenes")

# Histograma de intensidades de píxel (0-255)
plt.figure(figsize=(6, 4))
plt.hist(x_train.reshape(-1), bins=50)
plt.title("Distribución de intensidades de píxel (x_train)")
plt.xlabel("Valor de píxel (0-255)")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.show()
```

# **🧼 SECCIÓN 2 – Normalización y preparación de datos**

### 🔹 2.1 Normalizar imágenes y añadir canal

```python
# ============================================
# SECCIÓN 2: PREPROCESAMIENTO
# ============================================

# Convertir a float32 y normalizar [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("Rango de x_train después de normalizar:",
      x_train.min(), "a", x_train.max())
print("Rango de x_test después de normalizar:",
      x_test.min(), "a", x_test.max())

# Añadir dimensión de canal (gris -> 1 canal)
x_train = np.expand_dims(x_train, axis=-1)  # (N, 28, 28, 1)
x_test = np.expand_dims(x_test, axis=-1)

print("Nueva forma de x_train:", x_train.shape)
print("Nueva forma de x_test :", x_test.shape)
```

### 🔹 2.2 Visualización rápida post-normalización

```python
plt.figure(figsize=(6, 3))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(x_train[i].squeeze(), cmap="gray")
    plt.title(f"{class_names[y_train[i]]}")
    plt.axis("off")

plt.suptitle("Muestras normalizadas de Fashion-MNIST", fontsize=14)
plt.tight_layout()
plt.show()
```

# **🌫️ SECCIÓN 3 – Generación de imágenes ruidosas (Denoising)**

Vamos a generar una versión **ruidosa** de las imágenes agregando **ruido gaussiano** controlado.

### 🔹 3.1 Función para añadir ruido gaussiano

```python
# ============================================
# SECCIÓN 3: GENERACIÓN DE DATOS RUIDOSOS
# ============================================

def add_gaussian_noise(images, mean=0.0, std=0.3):
    """
    Añade ruido gaussiano a un conjunto de imágenes.
    - images: array de forma (N, H, W, C) con valores en [0, 1]
    - mean: media del ruido
    - std: desviación estándar del ruido
    
    Devuelve:
    - imágenes con ruido, recortadas a [0, 1]
    """
    noise = np.random.normal(loc=mean, scale=std, size=images.shape)
    noisy_images = images + noise
    # Recortar para mantener dentro de [0, 1]
    noisy_images = np.clip(noisy_images, 0., 1.)
    return noisy_images

# Generar conjuntos ruidosos
x_train_noisy = add_gaussian_noise(x_train, mean=0.0, std=0.4)
x_test_noisy = add_gaussian_noise(x_test, mean=0.0, std=0.4)

print("Rango de x_train_noisy:", x_train_noisy.min(), "a", x_train_noisy.max())
print("Rango de x_test_noisy :", x_test_noisy.min(), "a", x_test_noisy.max())
```

### 🔹 3.2 Visualizar imágenes limpias vs ruidosas

```python
num_show = 8
plt.figure(figsize=(16, 4))

for i in range(num_show):
    # Original
    ax = plt.subplot(2, num_show, i+1)
    plt.imshow(x_train[i].squeeze(), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # Ruidosa
    ax = plt.subplot(2, num_show, i+1+num_show)
    plt.imshow(x_train_noisy[i].squeeze(), cmap="gray")
    plt.title("Ruidosa")
    plt.axis("off")

plt.suptitle("Comparación de imágenes originales vs ruidosas", fontsize=16)
plt.tight_layout()
plt.show()
```

### 🔹 3.3 Histograma de intensidades con ruido

```python
plt.figure(figsize=(6, 4))
plt.hist(x_train_noisy.reshape(-1), bins=50)
plt.title("Distribución de intensidades con ruido (x_train_noisy)")
plt.xlabel("Valor de píxel (0-1)")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.show()
```

> 🔎 **Reflexión**: Observa cómo el ruido ensancha la distribución de intensidades y hace las imágenes visualmente más difíciles de interpretar.

# **🧪 SECCIÓN 4 – División en entrenamiento y validación**

```python
# ============================================
# SECCIÓN 4: SPLIT TRAIN / VALIDATION
# ============================================

val_fraction = 0.1
val_size = int(len(x_train) * val_fraction)

x_val_clean = x_train[:val_size]
x_val_noisy = x_train_noisy[:val_size]

x_train_clean = x_train[val_size:]
x_train_noisy_sub = x_train_noisy[val_size:]

print("x_train_clean     :", x_train_clean.shape)
print("x_train_noisy_sub :", x_train_noisy_sub.shape)
print("x_val_clean       :", x_val_clean.shape)
print("x_val_noisy       :", x_val_noisy.shape)
```

# **🧠 SECCIÓN 5 – Definición del Denoising Autoencoder Convolucional**

Arquitectura general:

- Entrada: imagen **ruidosa** (28x28x1).
- Salida: imagen **limpia** (28x28x1).
- Pérdida: distancia entre salida reconstruida y original limpia.

```python
# ============================================
# SECCIÓN 5: DEFINICIÓN DEL DENOISING AUTOENCODER
# ============================================

input_shape = (28, 28, 1)

denoise_inputs = layers.Input(shape=input_shape, name="denoise_input")

# Encoder
x = layers.Conv2D(32, (3,3), activation="relu", padding="same", name="enc_conv1")(denoise_inputs)
x = layers.MaxPooling2D((2,2), padding="same", name="enc_pool1")(x)

x = layers.Conv2D(64, (3,3), activation="relu", padding="same", name="enc_conv2")(x)
encoded = layers.MaxPooling2D((2,2), padding="same", name="enc_pool2")(x)

print("Forma del espacio latente (encoded):", encoded.shape)

# Decoder
x = layers.Conv2D(64, (3,3), activation="relu", padding="same", name="dec_conv1")(encoded)
x = layers.UpSampling2D((2,2), name="dec_up1")(x)

x = layers.Conv2D(32, (3,3), activation="relu", padding="same", name="dec_conv2")(x)
x = layers.UpSampling2D((2,2), name="dec_up2")(x)

denoise_outputs = layers.Conv2D(
    1, (3,3), activation="sigmoid", padding="same", name="dec_output"
)(x)

denoising_autoencoder = models.Model(
    denoise_inputs, denoise_outputs, name="denoising_autoencoder"
)

denoising_autoencoder.summary()
```

> ###### **⚠️ Advertencia técnica:**
>
> - Usamos `sigmoid` como activación final porque los píxeles están en [0, 1].
> - El modelo es undercomplete (reduce resolución a 7x7x64) para obligar a aprender características robustas.

# **⚙️ SECCIÓN 6 – Compilación y entrenamiento**

```python
# ============================================
# SECCIÓN 6: COMPILACIÓN Y ENTRENAMIENTO
# ============================================

denoising_autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy"
)

epochs = 20
batch_size = 256

history = denoising_autoencoder.fit(
    x_train_noisy_sub, x_train_clean,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(x_val_noisy, x_val_clean)
)
```

### 🔹 6.1 Curvas de pérdida (train vs val)

```python
plt.figure(figsize=(6, 4))
plt.plot(history.history["loss"], label="Pérdida de entrenamiento")
plt.plot(history.history["val_loss"], label="Pérdida de validación")
plt.title("Curvas de pérdida (Denoising Autoencoder)")
plt.xlabel("Época")
plt.ylabel("Pérdida (binary_crossentropy)")
plt.legend()
plt.grid(True)
plt.show()
```

> 💡 Si `val_loss` empieza a subir mientras `loss` baja, puede indicarse sobreajuste → puedes reducir epochs, añadir regularización, etc.

# **🧾 SECCIÓN 7 – Evaluación cualitativa: comparación visual**

##### Compararemos tres versiones:

1. Imagen limpia (ground truth).
2. Imagen ruidosa de entrada.
3. Imagen reconstruida (denoised).

```python
# ============================================
# SECCIÓN 7: EVALUACIÓN CUALITATIVA
# ============================================

# Obtener reconstrucciones para el conjunto de test
x_test_denoised = denoising_autoencoder.predict(x_test_noisy)

num_images = 10
plt.figure(figsize=(18, 6))

for i in range(num_images):
    # Fila 1: Limpia original
    ax = plt.subplot(3, num_images, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap="gray")
    plt.title("Limpia")
    plt.axis("off")
    
    # Fila 2: Ruidosa
    ax = plt.subplot(3, num_images, i + 1 + num_images)
    plt.imshow(x_test_noisy[i].squeeze(), cmap="gray")
    plt.title("Ruidosa")
    plt.axis("off")
    
    # Fila 3: Reconstruida
    ax = plt.subplot(3, num_images, i + 1 + 2*num_images)
    plt.imshow(x_test_denoised[i].squeeze(), cmap="gray")
    plt.title("Denoised")
    plt.axis("off")

plt.suptitle("Comparación Limpia vs Ruidosa vs Reconstruida (Denoising AE)", fontsize=16)
plt.tight_layout()
plt.show()
```

> ##### 🎯 Discusión:
>
> - ¿Qué detalles se recuperan bien?
> - ¿Qué tipo de ruido persiste?
> - ¿Se observa suavizado excesivo (blur)?

# **📏 SECCIÓN 8 – Evaluación cuantitativa: error de reconstrucción**

Usaremos **MSE** por imagen como un indicador de calidad de reconstrucción.

```python
# ============================================
# SECCIÓN 8: EVALUACIÓN CUANTITATIVA (MSE)
# ============================================

from sklearn.metrics import mean_squared_error

# Aplanar imágenes para cálculo de MSE por muestra
x_test_clean_flat = x_test.reshape((len(x_test), -1))
x_test_denoised_flat = x_test_denoised.reshape((len(x_test_denoised), -1))

mse_per_image = np.mean(
    np.power(x_test_clean_flat - x_test_denoised_flat, 2), axis=1
)

print("Forma de mse_per_image:", mse_per_image.shape)
print("Primeros 10 valores de MSE:", mse_per_image[:10])

print("\nEstadísticas del MSE:")
print("Mínimo :", mse_per_image.min())
print("Máximo :", mse_per_image.max())
print("Media  :", mse_per_image.mean())
print("Mediana:", np.median(mse_per_image))
```

### 🔹 8.1 Histograma del MSE

```python
plt.figure(figsize=(6, 4))
plt.hist(mse_per_image, bins=50)
plt.title("Distribución del error de reconstrucción (MSE) en test")
plt.xlabel("MSE por imagen")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.show()
```

> 🌐 En aplicaciones reales, imágenes con MSE inusualmente alto podrían considerarse **anómalas** (por ejemplo, defectos de fabricación, imágenes de otra clase, etc.).

# **🧩 SECCIÓN 9 – Consideraciones y advertencias técnicas**

Incluye estas reflexiones como celdas de texto en Colab:

```python
### Consideraciones y advertencias técnicas

- **Selección de ruido**:  
  - El tipo y la intensidad del ruido durante el entrenamiento deben parecerse al ruido real del problema.
  - Demasiado poco ruido → el modelo no se vuelve robusto.
  - Demasiado ruido → la reconstrucción se vuelve muy difícil.

- **Capacidad del modelo**:  
  - Más capas y filtros permiten mayor capacidad, pero aumentan costo computacional y riesgo de sobreajuste.

- **Tamaño del espacio latente**:  
  - En Denoising AE, un espacio latente razonablemente comprimido fuerza al modelo a aprender patrones robustos, no a copiar el ruido.

- **Métricas adicionales**:  
  - Podríamos usar PSNR (Peak Signal-to-Noise Ratio) además del MSE, y métricas perceptuales si se trabaja con imágenes complejas.

- **Generalización**:  
  - Probar con datasets distintos y tipos de ruido (poisson, speckle, blur) asegura que el autoencoder no se limite a un solo escenario.
```

# **✅ RESUMEN DE ESTA SEGUNDA PRÁCTICA**

En esta práctica has:

- Trabajado con **Fashion-MNIST**, un dataset abierto de imágenes de moda.
- Generado versiones ruidosas de las imágenes usando **ruido gaussiano**.
- Diseñado y entrenado un **Denoising Autoencoder convolucional**.
- Visualizado comparativamente imágenes limpias, ruidosas y reconstruidas.
- Calculado y analizado la distribución del **MSE por imagen**.
- Reflexionado sobre consideraciones prácticas en el uso de autoencoders para **reconstrucción y limpieza de imágenes**.



![pie](D:\001_Uniminuto2025\IMAGENES\pie.png)