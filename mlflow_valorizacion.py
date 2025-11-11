import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import mlflow
import mlflow.sklearn

# -----------------------------
# Cargar datos
# -----------------------------
data = pd.read_csv("datalimpio_modelo_final.csv")

# Filtrar outliers y preparar target
q_low = data['price_per_m2'].quantile(0.01)
q_high = data['price_per_m2'].quantile(0.99)
data_filtered = data[(data['price_per_m2'] >= q_low) & (data['price_per_m2'] <= q_high)].copy()
data_filtered['price_per_m2_log'] = np.log1p(data_filtered['price_per_m2'])

# Codificar zona
data_model = pd.get_dummies(data_filtered, columns=['zona'], drop_first=True)

# Variables predictoras y target
features_base = ['rooms', 'bedrooms', 'bathrooms', 'surface_total', 'mes_cont']
features_zonas = [c for c in data_model.columns if c.startswith('zona_')]
X = data_model[features_base + features_zonas]
y = data_model['price_per_m2_log']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Configurar MLflow apuntando al servidor local
# -----------------------------
mlflow.set_tracking_uri('http://localhost:5000')   # tu servidor MLflow local
mlflow.set_experiment("Valorizacion_Inmuebles")    # nombre del experimento

# -----------------------------
# Entrenar y registrar modelo
# -----------------------------
with mlflow.start_run():
    modelo = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=1,
        random_state=42
    )
    modelo.fit(X_train, y_train)

    # Predicciones y evaluación
    y_pred_log = modelo.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_real = np.expm1(y_test)
    mae = mean_absolute_error(y_test_real, y_pred)
    print(f"MAE: {mae:.2f} USD/m²")

    # Registrar parámetros, métricas y modelo
    mlflow.log_params({
        'n_estimators': 400,
        'max_depth': None,
        'min_samples_split': 5,
        'min_samples_leaf': 1
    })
    mlflow.log_metric("MAE", mae)
    mlflow.sklearn.log_model(modelo, "random_forest_model")

print("✅ Modelo registrado en MLflow. Abrí http://localhost:5000 para verlo")
