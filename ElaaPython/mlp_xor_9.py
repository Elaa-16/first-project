import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 1. Charger les données XOR depuis le fichier CSV
df = pd.read_csv('xor.csv')  # Le fichier doit contenir x1, x2 et label

# 2. Séparer les entrées (X) et les labels (y)
X = df[['x1', 'x2']]
y = df['label']

# 3. Définir le modèle MLP
mlp = MLPClassifier(
    hidden_layer_sizes=(4,),
    activation='relu',
    solver='adam',
    max_iter=3000,
    random_state=1,
    verbose=True
    # ⚠️ PAS de early_stopping ici car trop peu de données
)

# 4. Entraîner le modèle sur toutes les données
mlp.fit(X, y)

# 5. Évaluer le modèle sur les mêmes données (puisqu’on n’a pas de test set ici)
y_pred = mlp.predict(X)
print("\n✅ Accuracy:", accuracy_score(y, y_pred))
print("\n🧾 Classification Report:\n", classification_report(y, y_pred))

# 6. Tracer la courbe de perte
plt.plot(mlp.loss_curve_)
plt.title("Courbe de perte du MLP")
plt.xlabel("Itérations")
plt.ylabel("Perte (loss)")
plt.grid(True)
plt.show()
