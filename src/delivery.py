"""Carga, limpieza y tubería del modelo «¿Va a llegar tarde?».

Lo usan el notebook (src/explore.ipynb) y el script (src/train.py), así los dos
parten exactamente de los mismos datos y de la misma tubería.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RAIZ = Path(__file__).resolve().parents[1]
RUTA_DATOS = RAIZ / "data" / "raw" / "delivery_50000.csv"

SEMILLA = 42
OBJETIVO = "llego_tarde"

# Lo que se sabe del pedido cuando la app lo confirma.
CATEGORICAS = ["zona", "restaurante_id", "tipo_comida", "metodo_pago", "clima", "vehiculo"]
NUMERICAS = [
    "distancia_km", "cantidad_items", "importe", "cliente_nuevo", "pedidos_previos",
    "temperatura", "repartidores_disponibles", "pedidos_activos_zona",
    "tiempo_preparacion_estimado", "tiempo_prometido_min",
]
# Columnas que la tubería recibe tal cual llegan de la app.
ENTRADA = ["fecha_hora"] + CATEGORICAS + NUMERICAS
# Se conocen solo después de la entrega: no pueden ser características.
FUTURO = ["tiempo_real_min", "llego_tarde", "cancelado", "propina", "calificacion"]
# Derivadas de fecha_hora y de la carga de trabajo, se calculan dentro de la tubería.
DERIVADAS = ["hora", "dia_semana", "mes", "pedidos_por_repartidor", "margen_reparto_min"]

TEXTO = ["zona", "restaurante_id", "tipo_comida", "metodo_pago", "clima", "vehiculo"]
CODIGO_ERROR_TIEMPO = 999


def cargar(ruta=RUTA_DATOS):
    return pd.read_csv(ruta)


def limpiar(df):
    """Devuelve (pedidos limpios con objetivo conocido, recuento de lo que se quitó)."""
    df = df.copy()
    recuento = {"filas_originales": len(df)}

    for col in TEXTO:
        df[col] = df[col].str.strip().str.lower()
    df["restaurante_id"] = df["restaurante_id"].str.upper()

    # Repetidos: el mismo id_pedido. Uno solo difiere en «PIZZA»/«pizza», por eso se
    # normaliza el texto antes de buscarlos.
    recuento["pedidos_repetidos"] = int(df["id_pedido"].duplicated().sum())
    df = df.drop_duplicates("id_pedido", keep="first")

    # Valores imposibles en datos de entrada: se dejan vacíos y la tubería los imputa,
    # igual que haría con un pedido real que llegue mal.
    recuento["distancia_no_positiva"] = int((df["distancia_km"] <= 0).sum())
    df.loc[df["distancia_km"] <= 0, "distancia_km"] = np.nan
    recuento["importe_cero"] = int((df["importe"] <= 0).sum())
    df.loc[df["importe"] <= 0, "importe"] = np.nan

    # Sin objetivo fiable: cancelados (nunca se entregaron) y tiempo 999 (código de error,
    # su llego_tarde no sigue la regla del resto).
    recuento["cancelados"] = int(df["cancelado"].eq(1).sum())
    recuento["tiempo_999"] = int(df["tiempo_real_min"].eq(CODIGO_ERROR_TIEMPO).sum())
    df = df[df["cancelado"].eq(0) & df["tiempo_real_min"].ne(CODIGO_ERROR_TIEMPO)]
    df = df.dropna(subset=[OBJETIVO])
    df[OBJETIVO] = df[OBJETIVO].astype(int)

    recuento["filas_finales"] = len(df)
    return df.reset_index(drop=True), recuento


def anadir_derivadas(X):
    """Hora, día y mes del pedido, carga por repartidor y minutos de margen para el reparto."""
    X = X.copy()
    fecha = pd.to_datetime(X["fecha_hora"])
    X["hora"] = fecha.dt.hour
    X["dia_semana"] = fecha.dt.dayofweek
    X["mes"] = fecha.dt.month
    X["pedidos_por_repartidor"] = X["pedidos_activos_zona"] / X["repartidores_disponibles"]
    X["margen_reparto_min"] = X["tiempo_prometido_min"] - X["tiempo_preparacion_estimado"]
    return X.drop(columns="fecha_hora")


def tuberia(modelo, escalar=False):
    """Derivadas + imputación + one-hot + modelo, en un solo objeto que recibe ENTRADA."""
    numericas = [("imputar", SimpleImputer(strategy="median"))]
    if escalar:
        numericas.append(("escalar", StandardScaler()))
    preprocesado = ColumnTransformer([
        ("num", Pipeline(numericas), NUMERICAS + DERIVADAS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
    ])
    return Pipeline([
        ("derivadas", FunctionTransformer(anadir_derivadas)),
        ("preprocesado", preprocesado),
        ("modelo", modelo),
    ])
