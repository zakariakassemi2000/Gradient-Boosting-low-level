from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

# Charger les données Iris
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modèle CatBoost (verbose=0 pour éviter beaucoup de logs)
cat_model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=3, verbose=0)
cat_model.fit(X_train, y_train)

# Prédiction
y_pred = cat_model.predict(X_test)
print("Accuracy CatBoost:", accuracy_score(y_test, y_pred))
