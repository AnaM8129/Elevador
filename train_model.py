"""
train_model.py — Ejecuta UNA sola vez para generar model/mnist_fallback.pkl
Uso: python train_model.py
"""
import pathlib, pickle
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

pathlib.Path("model").mkdir(exist_ok=True)

print("Descargando MNIST…")
mnist  = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X      = mnist.data.astype("float32") / 255.0
y      = mnist.target.astype(int)

print("Entrenando (20 000 muestras, ~1-2 min)…")
clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=20, solver="adam", random_state=42, verbose=True)
clf.fit(X[:20000], y[:20000])

acc = accuracy_score(y[20000:22000], clf.predict(X[20000:22000]))
print(f"Precisión: {acc:.2%}")

with open("model/mnist_fallback.pkl", "wb") as f:
    pickle.dump(clf, f)
print("✅ Guardado en model/mnist_fallback.pkl")
