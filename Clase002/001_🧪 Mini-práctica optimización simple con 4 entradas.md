
![NeoDeepLab (1)](https://github.com/Grandote58/NeoDeepLab/blob/main/Clase002/assets/NeoDeepLab%20(1).png)

# **🧪 Mini-práctica: optimización simple con 4 entradas**

**Objetivo:** entrenar una red muy pequeña que prediga un valor de salida (por ejemplo, puntuación de calidad) a partir de **4 entradas** sintéticas.

> 📍Abre un cuaderno en Colab y ejecuta cada celda paso a paso.

### **🔹 Celda 1 – Datos sintéticos**

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# 100 ejemplos con 4 entradas
np.random.seed(42)
X = np.random.rand(100, 4)
# Relación lineal con algo de ruido
y = 3*X[:,0] + 2*X[:,1] - 1.5*X[:,2] + 0.5*X[:,3] + np.random.randn(100)*0.2

# Normalizar
from sklearn.preprocessing import StandardScaler
scaler_X, scaler_y = StandardScaler(), StandardScaler()
X_s = scaler_X.fit_transform(X)
y_s = scaler_y.fit_transform(y.reshape(-1,1))
```

### **🔹 Celda 2 – Modelo y compilación**

```python
model = keras.Sequential([
    keras.layers.Input(shape=(4,)),        # 4 entradas
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])

model.compile(
    loss='mse',
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    metrics=['mae']
)
model.summary()
```

### **🔹 Celda 3 – Entrenamiento con Early Stopping**

```python
early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_s, y_s,
    validation_split=0.2,
    epochs=200,
    batch_size=16,
    callbacks=[early],
    verbose=0
)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Curva de pérdida'); plt.legend(); plt.show()
```

### **🔹 Celda 4 – Evaluación**

```
loss, mae = model.evaluate(X_s, y_s, verbose=0)
print(f"✅ MSE: {loss:.4f} | MAE: {mae:.4f}")
```

### **💡 Qué observar:**

- Si el **learning_rate** es muy alto, la pérdida oscila.
- Si es muy bajo, tarda en converger.
- Puedes probar: `0.001`, `0.005`, `0.02` y ver el impacto.
- Cambia `batch_size` (8, 16, 32) para comparar estabilidad y velocidad.

## Reto guiado 🎯

Modifica **un solo hiperparámetro** y comenta:

- Cómo cambia la curva de pérdida.
- Cuál configuración alcanzó menor error.
- Qué harías para optimizar sin probar manualmente (pista: *Grid Search* o *Random Search*).

## Resumen 🧩

| Concepto                   | Clave                                                       |
| -------------------------- | ----------------------------------------------------------- |
| **Descenso del gradiente** | Método numérico que ajusta los pesos minimizando la pérdida |
| **Learning rate**          | Controla la velocidad del aprendizaje                       |
| **Hiperparámetros**        | Configuraciones que afectan cómo aprende la red             |
| **Optimización**           | Búsqueda del conjunto que logra mejor rendimiento           |
| **Early Stopping**         | Detiene el entrenamiento cuando deja de mejorar             |




![PiePagina](Clase002/assets/PiePagina-1761060152178-4.png)

