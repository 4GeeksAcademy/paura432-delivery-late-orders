# ¿Va a llegar tarde el pedido?

Actividad «Tu modelo, del notebook a train.py» del bootcamp AI Engineering de 4Geeks, hecha a partir del template [`machine-learning-python-template`](https://github.com/4GeeksAcademy/machine-learning-python-template).

Con unos 50.000 pedidos de una app de delivery (enero de 2024 a junio de 2025), el modelo predice **en el momento de confirmar el pedido** si va a llegar más de 5 minutos después de lo prometido. Es una clasificación binaria: llega tarde el 25,5 % de los pedidos entregados.

## Resultados

- **Limpieza.**
  - Se quitan 180 pedidos repetidos, uno de ellos solo distinto por «PIZZA»/«pizza».
  - Se normalizan las categorías en mayúsculas.
  - Las distancias negativas y los importes de 0 € pasan a vacíos, que imputa la tubería.
  - Se descartan los cancelados y los pedidos con el código de error `tiempo_real_min = 999`.
  - Quedan 46.875 pedidos.
- **Comparación** con validación cruzada de 5 folds y precisión media (AP) como métrica:
  - línea base 0,255;
  - HistGradientBoosting 0,801 y XGBoost 0,793;
  - random forest 0,783 y regresión logística 0,775;
  - árbol de decisión 0,731.
- **Ajuste.** Búsqueda aleatoria de 30 combinaciones y después una cuadrícula de 24. XGBoost pasa de 0,793 ± 0,005 a 0,806 ± 0,006.
- **Prueba**, medida una vez:

  | | Línea base | Modelo |
  |---|---|---|
  | AP | 0,255 | 0,813 |
  | ROC AUC | — | 0,912 |
  | Exactitud | 0,745 | 0,849 |

  Con el umbral de 0,36 detecta 1.768 de los 2.391 retrasos (recall 0,74, precisión 0,69). Donde más se equivoca es con probabilidades intermedias, en bici y con lluvia o tormenta.

## Contenido

| Ruta | Qué es |
| --- | --- |
| `src/explore.ipynb` | Notebook ejecutado con las partes 1 a 8 y sus respuestas en celdas de texto |
| `NOTAS.md` | Respuestas de las partes 9 a 14 y del extra |
| `src/delivery.py` | Limpieza, columnas y tubería; la usan el notebook y el script |
| `src/train.py` | Entrena, aplica la barrera y guarda la versión indicada con `--version` |
| `src/app.py` | Extra: carga el modelo una vez, valida pedidos y predice |
| `data/raw/delivery_50000.csv` | Dataset de la actividad, sin cambios |
| `models/llegada_tarde_v1.joblib` | Tubería completa entrenada con todos los pedidos |
| `models/llegada_tarde_v1.json` | Qué hay en la v1: hiperparámetros, columnas, categorías, umbral, métricas y huella de los datos |
| `requirements.txt` | Dependencias del template más `joblib`, todas con versión fija |

El resto (`.devcontainer/`, `src/utils.py`, `.env.example`, `data/interim/`, `data/processed/`) viene del template y este proyecto no lo usa.

## Ejecución

En Codespaces el entorno instala `requirements.txt` al crearse. En local:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Entrenar una versión nueva (no sobrescribe las existentes y no guarda si la precisión media en la prueba baja de 0,78):

```sh
python src/train.py --version v2
```

Predecir con pedidos inventados o con un JSON propio:

```sh
python src/app.py
python src/app.py pedidos.json --version v1
```

El notebook tarda unos 30 minutos en ejecutarse entero, sobre todo por las búsquedas de hiperparámetros. Los scripts limitan OpenMP a 8 hilos (`OMP_NUM_THREADS`): en la máquina donde se hizo, con 16, el entrenamiento se quedaba atascado.
