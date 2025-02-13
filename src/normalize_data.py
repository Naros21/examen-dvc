import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

scaler = StandardScaler()
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
if isinstance(X_train.iloc[:, 0][0], str):
    X_train = X_train.iloc[:, 1:]
    X_test = X_test.iloc[:, 1:]

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pd.DataFrame(X_train_scaled).to_csv("data/processed/X_train_scaled.csv", index=False)
pd.DataFrame(X_test_scaled).to_csv("data/processed/X_test_scaled.csv", index=False)
