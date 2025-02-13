import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib

X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv")
params = {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}

model = GridSearchCV(RandomForestRegressor(), params, cv=5)
model.fit(X_train, y_train.values.ravel())
joblib.dump(model.best_params_, "models/best_params.pkl")
