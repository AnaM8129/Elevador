import pathlib, pickle
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier

pathlib.Path("model").mkdir(exist_ok=True)

print("Descargando MNIST…")
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X = mnist.data.astype("float32") / 255.0
y = mnist.target.astype(int)

print("Entrenando KNN (instantáneo)…")
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X[:5000], y[:5000])   # solo 5000 muestras, suficiente para dígitos 1-6

with open("model/mnist_fallback.pkl", "wb") as f:
    pickle.dump(clf, f)

print("✅ Listo en model/mnist_fallback.pkl")
