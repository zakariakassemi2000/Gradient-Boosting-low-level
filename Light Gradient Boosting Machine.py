from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score

# Charger les données Iris
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modèle LightGBM
lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
lgb_model.fit(X_train, y_train)

# Prédiction
y_pred = lgb_model.predict(X_test)
print("Accuracy LightGBM:", accuracy_score(y_test, y_pred))
