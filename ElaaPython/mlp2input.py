import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load your 3-input XOR data from xor_3.csv
df = pd.read_csv("xor_2.csv")

X = df.iloc[:, :-1].values  # all columns except last are inputs
y = df.iloc[:, -1].values   # last column is output

# Create and train MLP
mlp = MLPClassifier(hidden_layer_sizes=(3,), activation='tanh', max_iter=2000, random_state=42)
mlp.fit(X, y)

# Predict on the training data
preds = mlp.predict(X)
acc = accuracy_score(y, preds)

print("Predictions:", preds)
print(f"Accuracy on training data: {acc:.2f}")