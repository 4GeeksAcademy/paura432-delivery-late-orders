"""Extra: usa el modelo guardado para avisar de pedidos que probablemente lleguen tarde.

Uso:
    python src/app.py                        # pedidos inventados de ejemplo
    python src/app.py pedidos.json           # lista de pedidos en JSON
    python src/app.py --version v1 pedidos.json

Carga models/llegada_tarde_<version>.joblib una sola vez y valida cada pedido
contra su .json antes de predecir.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")

import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import delivery  # noqa: E402  (la tubería guardada usa delivery.anadir_derivadas)

EJEMPLOS = [
    {"fecha_hora": "2025-07-04 21:15", "zona": "Centro", "restaurante_id": "R073", "tipo_comida": "pizza",
     "metodo_pago": "app", "clima": "lluvia", "vehiculo": "bici", "distancia_km": 4.8, "cantidad_items": 3,
     "importe": 38.5, "cliente_nuevo": 0, "pedidos_previos": 12, "temperatura": 14.0,
     "repartidores_disponibles": 6, "pedidos_activos_zona": 41, "tiempo_preparacion_estimado": 22,
     "tiempo_prometido_min": 35},
    {"fecha_hora": "2025-07-05 13:40", "zona": "Norte", "restaurante_id": "R020", "tipo_comida": "saludable",
     "metodo_pago": "tarjeta", "clima": "despejado", "vehiculo": "moto", "distancia_km": 1.2, "cantidad_items": 1,
     "importe": 12.9, "cliente_nuevo": 1, "pedidos_previos": 0, "temperatura": 27.5,
     "repartidores_disponibles": 22, "pedidos_activos_zona": 9, "tiempo_preparacion_estimado": 12,
     "tiempo_prometido_min": 35},
    {"fecha_hora": "2025-07-05 20:05", "zona": "Marte", "restaurante_id": "R020", "tipo_comida": "pizza",
     "metodo_pago": "tarjeta", "clima": "despejado", "vehiculo": "moto", "distancia_km": 2.0, "cantidad_items": 2,
     "importe": 20.0, "cliente_nuevo": 0, "pedidos_previos": 5, "temperatura": 20.0,
     "repartidores_disponibles": 15, "pedidos_activos_zona": 20, "tiempo_preparacion_estimado": 15,
     "tiempo_prometido_min": 35},
]


class PedidoInvalido(ValueError):
    pass


def cargar(version):
    ruta = delivery.RAIZ / "models" / f"llegada_tarde_{version}.joblib"
    info = json.loads(ruta.with_suffix(".json").read_text(encoding="utf-8"))
    return joblib.load(ruta), info


def validar(pedido, info):
    """Devuelve el pedido como fila de la tubería o lanza PedidoInvalido."""
    faltan = [c for c in info["entrada"] if c not in pedido]
    if faltan:
        raise PedidoInvalido(f"faltan campos: {', '.join(faltan)}")
    fila = {c: pedido[c] for c in info["entrada"]}
    for col, validos in info["categorias"].items():
        valor = str(fila[col]).strip()
        valor = valor.upper() if col == "restaurante_id" else valor.lower()
        if valor not in validos:
            raise PedidoInvalido(f"valor desconocido en {col}: {fila[col]!r}")
        fila[col] = valor
    for col in delivery.NUMERICAS:
        valor = fila[col]
        if not isinstance(valor, (int, float)) or isinstance(valor, bool) or math.isnan(valor):
            raise PedidoInvalido(f"{col} debe ser un número: {valor!r}")
    if fila["distancia_km"] <= 0 or fila["importe"] <= 0 or fila["repartidores_disponibles"] <= 0:
        raise PedidoInvalido("distancia, importe y repartidores deben ser positivos")
    try:
        pd.to_datetime(fila["fecha_hora"])
    except (ValueError, TypeError) as error:
        raise PedidoInvalido(f"fecha_hora no válida: {fila['fecha_hora']!r}") from error
    return fila


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pedidos", nargs="?", help="JSON con una lista de pedidos")
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()

    modelo, info = cargar(args.version)
    pedidos = json.loads(Path(args.pedidos).read_text(encoding="utf-8")) if args.pedidos else EJEMPLOS
    umbral = info["umbral_decision"]
    print(f"Modelo {info['archivo_modelo']} · umbral {umbral}")
    for i, pedido in enumerate(pedidos, 1):
        try:
            fila = validar(pedido, info)
        except PedidoInvalido as error:
            print(f"Pedido {i}: rechazado, {error}")
            continue
        proba = float(modelo.predict_proba(pd.DataFrame([fila]))[0, 1])
        aviso = "TARDE" if proba >= umbral else "a tiempo"
        print(f"Pedido {i}: {aviso} (probabilidad de retraso {proba:.2f})")


if __name__ == "__main__":
    main()
