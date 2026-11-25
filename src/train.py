"""Entrena el modelo «¿Va a llegar tarde?» y lo guarda versionado en models/.

Uso:
    python src/train.py --version v1

Repite el examen final del notebook (misma limpieza, mismo split, mismos
hiperparámetros), se niega a guardar si el modelo no supera la barrera y, si la
supera, reentrena con todos los pedidos y guarda la tubería entera junto a un
.json que describe qué hay dentro. Nunca sobrescribe una versión existente.
"""

import os

# Limita los hilos de OpenMP: con los 16 de esta máquina (WSL) el entrenamiento se atasca.
os.environ.setdefault("OMP_NUM_THREADS", "8")

import argparse
import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import delivery  # noqa: E402

CARPETA_MODELOS = delivery.RAIZ / "models"
NOMBRE = "llegada_tarde"

# Hiperparámetros elegidos en el notebook (paso 7): mejor combinación de la cuadrícula.
HIPERPARAMETROS = {
    "n_estimators": 550,
    "learning_rate": 0.04,
    "max_depth": 4,
    "min_child_weight": 15,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
}

# Barrera (paso 12): precisión media mínima en la prueba. 0,78 es lo que ya consigue una
# regresión logística sin ajustar; un XGBoost ajustado que no la supere indica un problema.
UMBRAL_AP = 0.78


def modelo_final():
    return delivery.tuberia(XGBClassifier(n_jobs=8, random_state=delivery.SEMILLA, **HIPERPARAMETROS))


def umbral_decision(tuberia, X, y):
    """Umbral de probabilidad que maximiza F1 en predicciones fuera de fold del entrenamiento."""
    cv = StratifiedKFold(5, shuffle=True, random_state=delivery.SEMILLA)
    proba = cross_val_predict(tuberia, X, y, cv=cv, method="predict_proba")[:, 1]
    candidatos = np.round(np.arange(0.05, 0.96, 0.01), 2)
    f1 = [f1_score(y, proba >= u) for u in candidatos]
    return float(candidatos[int(np.argmax(f1))])


def metricas(y, proba, umbral):
    pred = proba >= umbral
    return {
        "average_precision": round(float(average_precision_score(y, proba)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "f1": round(float(f1_score(y, pred)), 4),
        "recall": round(float(recall_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred)), 4),
    }


def huella(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", required=True, help="Etiqueta de la versión, por ejemplo v1")
    parser.add_argument("--umbral-ap", type=float, default=UMBRAL_AP,
                        help=f"Precisión media mínima en la prueba (por defecto {UMBRAL_AP})")
    args = parser.parse_args()

    if not re.fullmatch(r"v\d+", args.version):
        parser.error("--version debe tener la forma v1, v2, ...")
    ruta_modelo = CARPETA_MODELOS / f"{NOMBRE}_{args.version}.joblib"
    ruta_info = ruta_modelo.with_suffix(".json")
    if ruta_modelo.exists() or ruta_info.exists():
        sys.exit(f"Ya existe la versión {args.version} en {CARPETA_MODELOS}: usa otra, no se sobrescribe.")

    datos, limpieza = delivery.limpiar(delivery.cargar())
    X, y = datos[delivery.ENTRADA], datos[delivery.OBJETIVO]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=delivery.SEMILLA)

    # Examen final: lo mismo que el paso 8 del notebook.
    umbral = umbral_decision(modelo_final(), X_train, y_train)
    tuberia = modelo_final().fit(X_train, y_train)
    prueba = metricas(y_test, tuberia.predict_proba(X_test)[:, 1], umbral)
    print(f"Prueba ({len(y_test)} pedidos): {prueba}")
    print(f"Umbral de decisión: {umbral}")

    if prueba["average_precision"] < args.umbral_ap:
        sys.exit(f"No se guarda el modelo: precisión media {prueba['average_precision']} "
                 f"por debajo de la barrera {args.umbral_ap}.")

    # Reentrenar con todos los pedidos y guardar la tubería entera.
    final = clone(tuberia).fit(X, y)
    CARPETA_MODELOS.mkdir(exist_ok=True)
    joblib.dump(final, ruta_modelo)
    info = {
        "version": args.version,
        "creado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pregunta": "¿Va a llegar tarde el pedido? Predicción al confirmar el pedido.",
        "objetivo": delivery.OBJETIVO,
        "modelo": "XGBClassifier",
        "hiperparametros": HIPERPARAMETROS,
        "entrada": delivery.ENTRADA,
        "derivadas": delivery.DERIVADAS,
        # Valores vistos en el entrenamiento: una app debe rechazar los demás (ver src/app.py).
        "categorias": {col: sorted(map(str, X[col].dropna().unique())) for col in delivery.CATEGORICAS},
        "umbral_decision": umbral,
        "barrera_average_precision": args.umbral_ap,
        "metricas_prueba": prueba,
        "pedidos_prueba": int(len(y_test)),
        "pedidos_entrenamiento_final": int(len(y)),
        "proporcion_tarde": round(float(y.mean()), 4),
        "limpieza": limpieza,
        "datos": {"archivo": delivery.RUTA_DATOS.name, "sha256": huella(delivery.RUTA_DATOS)},
        "entorno": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                    "xgboost": xgboost.__version__},
        "archivo_modelo": ruta_modelo.name,
    }
    ruta_info.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Modelo guardado en {ruta_modelo.relative_to(delivery.RAIZ)} "
          f"({ruta_modelo.stat().st_size} bytes) y descripción en {ruta_info.relative_to(delivery.RAIZ)}")


if __name__ == "__main__":
    main()
