
![NeoDeepLab (1)](https://github.com/Grandote58/NeoDeepLab/blob/main/Clase002/assets/NeoDeepLab%20(1).png)


# **🧪 Mini-práctica: medir rendimiento de dos modelos de regresión**

**Objetivo:** comparar la eficiencia de dos modelos (uno lineal y uno no lineal) usando 4 variables de entrada.

> 📍Ejecuta este código paso a paso en Google Colab.

### **🔹 Celda 1 – Datos sintéticos**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import time

# Datos aleatorios con 4 características
np.random.seed(42)
X = np.random.rand(500, 4)
# Relación no lineal con algo de ruido
y = 3*X[:,0] + 2*np.square(X[:,1]) - 1.5*X[:,2] + np.sin(2*np.pi*X[:,3]) + np.random.randn(500)*0.2

# Dividir y escalar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```

### **🔹 Celda 2 – Entrenar dos modelos y medir tiempos**

```python
# Modelo lineal
start = time.time()
lin = LinearRegression().fit(X_train_s, y_train)
t_lin = time.time() - start

# Red neuronal pequeña
start = time.time()
mlp = MLPRegressor(hidden_layer_sizes=(8,4), activation='relu',
                   solver='adam', learning_rate_init=0.01,
                   max_iter=1000, random_state=42)
mlp.fit(X_train_s, y_train)
t_mlp = time.time() - start

print(f"⏱ Tiempo LR: {t_lin:.4f}s | Tiempo MLP: {t_mlp:.4f}s")
```

### **🔹 Celda 3 – Evaluación y comparación**

```python
def report(model, X_t, y_t, name):
    y_pred = model.predict(X_t)
    mae = mean_absolute_error(y_t, y_pred)
    mse = mean_squared_error(y_t, y_pred)
    r2 = r2_score(y_t, y_pred)
    print(f"\n📊 {name}")
    print(f"MAE: {mae:.3f} | MSE: {mse:.3f} | R²: {r2:.3f}")
    return mae, mse, r2

mae_lin, mse_lin, r2_lin = report(lin, X_test_s, y_test, "Regresión Lineal")
mae_mlp, mse_mlp, r2_mlp = report(mlp, X_test_s, y_test, "Red Neuronal (MLP)")
```

### **🔹 Celda 4 – Visualización comparativa**

```
labels = ['MAE', 'MSE', 'R2']
lr_scores = [mae_lin, mse_lin, r2_lin]
mlp_scores = [mae_mlp, mse_mlp, r2_mlp]

plt.figure(figsize=(6,4))
plt.bar(labels, lr_scores, alpha=0.6, label='Lineal')
plt.bar(labels, mlp_scores, alpha=0.6, label='MLP')
plt.title("Comparación de medidas de rendimiento")
plt.legend()
plt.show()
```

### **💡 Qué observar:**

- ¿Cuál modelo tiene menor **MSE** y **MAE**?
- ¿Cuál explica mejor la variabilidad (**R²**)?
- ¿Cuál tarda menos en entrenar (tiempo)?
- ¿Existe un equilibrio entre **precisión** y **eficiencia computacional**?

## 6️⃣ Reto guiado 🎯

1. Cambia el número de neuronas en `hidden_layer_sizes` y observa cómo cambia el **tiempo** y el **error**.
2. Ajusta el `learning_rate_init` (por ejemplo `0.001`, `0.02`) y analiza si el modelo converge más rápido o se vuelve inestable.
3. Reflexiona: ¿qué modelo elegirías si trabajas con datos en tiempo real?

## 7️⃣ Resumen ⚙️

| Concepto                     | Qué significa                                | Por qué importa                         |
| ---------------------------- | -------------------------------------------- | --------------------------------------- |
| **Medidas de rendimiento**   | Números que reflejan precisión del modelo    | Evalúan calidad del aprendizaje         |
| **Eficiencia**               | Relación entre rendimiento y recursos usados | Afecta escalabilidad en sistemas reales |
| **MAE / MSE / R²**           | Errores y bondad de ajuste en regresión      | Permiten comparar modelos               |
| **Balance precisión-tiempo** | Buen modelo ≠ modelo lento                   | Se busca equilibrio según la aplicación |



![PiePagina](https://github.com/Grandote58/NeoDeepLab/blob/main/Clase002/assets/PiePagina-1761060152178-4.png)




