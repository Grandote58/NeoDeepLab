
![NeoDeepLab (1)](https://github.com/Grandote58/NeoDeepLab/blob/main/Clase002/assets/NeoDeepLab%20(1).png)

# **🧪 Mini-práctica: comparar dos modelos de clasificación**

**Objetivo:** usar datos sintéticos con **4 entradas** para evaluar el rendimiento de dos modelos:
 uno lineal (Logistic Regression) y uno no lineal (Random Forest).

### **🔹 Celda 1 – Datos sintéticos**

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 500 muestras, 4 características
X, y = make_classification(n_samples=500, n_features=4, n_informative=3,
                           n_redundant=0, random_state=42)

# División y escalado
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
```

### **🔹 Celda 2 – Entrenar modelos y predecir**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Modelos
log_reg = LogisticRegression(max_iter=1000, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

log_reg.fit(X_train_s, y_train)
rf.fit(X_train_s, y_train)

y_pred_lr = log_reg.predict(X_test_s)
y_pred_rf = rf.predict(X_test_s)
```

### **🔹 Celda 3 – Evaluar métricas de rendimiento**

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

def evaluar(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"\n📊 {model_name}")
    print(f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")
    return acc, prec, rec, f1

m1 = evaluar(y_test, y_pred_lr, "Regresión Logística")
m2 = evaluar(y_test, y_pred_rf, "Random Forest")

# Matriz de confusión
fig, ax = plt.subplots(1,2, figsize=(10,4))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr, ax=ax[0], cmap="Blues")
ax[0].set_title("Logistic Regression")
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf, ax=ax[1], cmap="Greens")
ax[1].set_title("Random Forest")
plt.show()
```

### **🔹 Celda 4 – AUC-ROC y comparación visual**

```python
from sklearn.metrics import roc_curve

# Probabilidades para AUC
y_prob_lr = log_reg.predict_proba(X_test_s)[:,1]
y_prob_rf = rf.predict_proba(X_test_s)[:,1]

auc_lr = roc_auc_score(y_test, y_prob_lr)
auc_rf = roc_auc_score(y_test, y_prob_rf)

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.plot(fpr_lr, tpr_lr, label=f"LR (AUC={auc_lr:.2f})")
plt.plot(fpr_rf, tpr_rf, label=f"RF (AUC={auc_rf:.2f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Curva ROC")
plt.legend()
plt.show()
```

### **💡 Qué analizar**

- ¿Qué modelo logra mejor **precision** y **recall**?
- ¿Cuál tiene mejor **balance global (F1)**?
- ¿El **AUC** muestra buena separación entre clases?
- ¿Qué tan distintas son sus **matrices de confusión**?

## 5️⃣ Reto guiado 🎯

1. Cambia el número de **árboles** del `RandomForestClassifier` (`n_estimators=50,200`).
2. Observa cómo cambia el **tiempo de ejecución** y las métricas.
3. Prueba distintos **criterios** (`criterion='entropy'` o `'gini'`).
4. Reflexiona: ¿qué métrica usarías si tu aplicación no puede tolerar falsos negativos?

## 6️⃣ Resumen express 🧩

| Métrica       | Evalúa                              | Ideal cuando...           |
| ------------- | ----------------------------------- | ------------------------- |
| **Accuracy**  | % aciertos globales                 | Clases balanceadas        |
| **Precision** | Exactitud de los positivos          | Falsos positivos costosos |
| **Recall**    | Cobertura de los positivos          | Falsos negativos costosos |
| **F1-Score**  | Equilibrio entre precisión y recall | Necesitas balance general |
| **AUC-ROC**   | Capacidad de separación             | Modelos probabilísticos   |



![PiePagina](https://github.com/Grandote58/NeoDeepLab/blob/main/Clase002/assets/PiePagina-1761060152178-4.png)

