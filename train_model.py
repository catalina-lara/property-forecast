# train_model.py
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

BASE_DIR = r"C:\Users\jorge.marin\NODRIVE\COURSERA\2025-5 Despliegue\Proyecto\Entrega2-jmm"
PATH_DATA = os.path.join(BASE_DIR, "datalimpio_modelo_final.csv")
OUTDIR = os.path.join(BASE_DIR, "model")
os.makedirs(OUTDIR, exist_ok=True)

# 1. Cargar datos
data = pd.read_csv(PATH_DATA)
# filtrar outliers tal como en notebook
q_low = data['price_per_m2'].quantile(0.01)
q_high = data['price_per_m2'].quantile(0.99)
data_filtered = data[(data['price_per_m2'] >= q_low) & (data['price_per_m2'] <= q_high)].copy()

# target log
data_filtered['price_per_m2_log'] = np.log1p(data_filtered['price_per_m2'])

# guardar zonas únicas
zonas_unicas = sorted(data_filtered['zona'].astype(str).unique())
pickle.dump(zonas_unicas, open(os.path.join(OUTDIR, "zonas_unicas.pkl"), "wb"))

# one-hot encoding (drop_first=True para replicar notebook)
data_model = pd.get_dummies(data_filtered, columns=['zona'], drop_first=True)

# features
features_base = ['rooms', 'bedrooms', 'bathrooms', 'surface_total', 'mes_cont']
features_zonas = [c for c in data_model.columns if c.startswith('zona_')]
X = data_model[features_base + features_zonas]
y = data_model['price_per_m2_log']

# train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# grid y entrenamiento
param_grid = {
    'n_estimators': [200, 500],
    'max_depth': [10, 15, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 3]
}
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Mejores parámetros:", grid_search.best_params_)

# evaluación (transformar de vuelta)
y_pred_log = best_model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_real = np.expm1(y_test)
mae = mean_absolute_error(y_test_real, y_pred)
print(f"MAE final: {mae:.2f} USD/m²")

# guardar artefactos
pickle.dump(best_model, open(os.path.join(OUTDIR, "best_model.pkl"), "wb"))
pickle.dump(list(X.columns), open(os.path.join(OUTDIR, "model_columns.pkl"), "wb"))

print("Modelo y artefactos guardados en:", OUTDIR)
