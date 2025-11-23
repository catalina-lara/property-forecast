# predict.py
import os
import pandas as pd
import numpy as np
import pickle

BASE_DIR = r"C:\Users\jorge.marin\NODRIVE\COURSERA\2025-5 Despliegue\Proyecto\Entrega2-jmm"
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
COLUMNS_PATH = os.path.join(MODEL_DIR, "model_columns.pkl")
ZONAS_PATH = os.path.join(MODEL_DIR, "zonas_unicas.pkl")

# cargar artefactos
_model = None
_model_columns = None
_zonas_unicas = None

def _load_artifacts():
    global _model, _model_columns, _zonas_unicas
    if _model is None:
        _model = pickle.load(open(MODEL_PATH, "rb"))
    if _model_columns is None:
        _model_columns = pickle.load(open(COLUMNS_PATH, "rb"))
    if _zonas_unicas is None:
        _zonas_unicas = pickle.load(open(ZONAS_PATH, "rb"))
    return _model, _model_columns, _zonas_unicas

def predict_batch(df_in):
    """
    df_in: dataframe que debe contener al menos:
      - 'zona' (str), 'rooms','bedrooms','bathrooms','surface_total','mes_cont'
    Devuelve: df_out = df_in.copy() con columna 'price_per_m2_pred' (USD/m2)
    """
    model, model_columns, zonas_unicas = _load_artifacts()
    df = df_in.copy()
    # asegurar tipos
    for c in ['rooms','bedrooms','bathrooms','surface_total','mes_cont']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        else:
            df[c] = 0

    df['zona'] = df['zona'].astype(str)

    # construir dummies como en entrenamiento (drop_first=True)
    dummies = pd.get_dummies(df['zona'], prefix='zona')
    # cuando en entrenamiento se usó drop_first=True, la columna para la primera zona no existe en X.
    # Lo importante es garantizar que todas las columnas en model_columns estén presentes en el mismo orden.
    X_tmp = df[['rooms','bedrooms','bathrooms','surface_total','mes_cont']].copy()
    # concatenar dummies
    X_tmp = pd.concat([X_tmp, dummies], axis=1)

    # Asegurar que todas las columnas esperadas estén presentes
    for col in model_columns:
        if col not in X_tmp.columns:
            X_tmp[col] = 0.0

    # ordenar columnas como el modelo
    X_tmp = X_tmp[model_columns]

    # predecir (modelo entrenado sobre log1p)
    pred_log = model.predict(X_tmp)
    pred = np.expm1(pred_log)  # volver a USD/m2

    df_out = df.copy()
    df_out['price_per_m2_pred'] = pred
    # precio total pronosticado
    df_out['price_pred_tmp'] = df_out['price_per_m2_pred'] * df_out['surface_total']
    return df_out
