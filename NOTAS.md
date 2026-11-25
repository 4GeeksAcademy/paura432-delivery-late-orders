# Notas: del notebook a `train.py` (partes 9 a 14)

## 9 · Qué pasa al script

**Pasa al script** solo lo necesario para volver a fabricar el modelo elegido:
- la carga;
- la limpieza (`delivery.limpiar`);
- el mismo split 80/20 con semilla 42;
- la tubería con los hiperparámetros finales;
- el umbral de decisión;
- el examen en la prueba, la barrera, el reentrenamiento con todos los pedidos y el guardado versionado.

**Se queda en el notebook** lo que sirvió para decidir: la exploración, los gráficos, la comparación de seis modelos, las búsquedas aleatorias y la cuadrícula, y el análisis de errores. Eso se ejecuta una vez para elegir; repetirlo en cada entrenamiento costaría unos 30 minutos y no cambiaría la decisión.

**Dónde vive cada cosa:**
- La limpieza y la tubería están en `src/delivery.py`. Las importan el notebook y el script, así que no hay dos copias que puedan separarse.
- Los hiperparámetros finales están escritos en `train.py` (`HIPERPARAMETROS`). El notebook los lee de ahí para comprobarlos.

## 10 · El primer `train.py`

Sí, da exactamente lo mismo en la prueba que el paso 8 del notebook:

```text
$ python src/train.py --version v1
Prueba (9375 pedidos): {'average_precision': 0.8132, 'roc_auc': 0.9122, 'f1': 0.7138, 'recall': 0.7394, 'precision': 0.6898}
Umbral de decisión: 0.36
Modelo guardado en models/llegada_tarde_v1.joblib (833001 bytes) y descripción en models/llegada_tarde_v1.json
```

El notebook da AP 0,8132, ROC AUC 0,9122, F1 0,7138, recall 0,7394, precisión 0,6898 y umbral 0,36. Coincide porque los dos usan la misma limpieza, el mismo split, la misma semilla y la misma función `modelo_final()`.

## 11 · Guardar

**Por qué se reentrena con todos los datos.** La prueba ya cumplió su función: medir una vez cómo se comporta el modelo con pedidos que no ha visto. A partir de ahí, dejar fuera el 20 % solo le quitaría ejemplos. El modelo que se guarda se entrena con los 46.875 pedidos, con los mismos hiperparámetros y el mismo umbral. Sus métricas son las de la prueba; no se pueden volver a medir porque ya no queda ningún pedido sin ver.

**Por qué se guarda la tubería entera y no solo el modelo.** XGBoost solo entiende la matriz numérica ya preparada. La tubería guarda además:
- el cálculo de hora, día, mes, pedidos por repartidor y margen;
- las medianas para imputar los vacíos;
- las categorías del one-hot, en el mismo orden que en el entrenamiento.

Con la tubería, quien use el modelo le pasa el pedido tal como lo tiene la app y no tiene que repetir (ni equivocarse en) el preprocesado.

Para cargarla hace falta que `src/delivery.py` sea importable, porque la tubería usa su función `anadir_derivadas`.

## 12 · La barrera

El script no guarda nada si la precisión media (AP) en la prueba baja de **0,78** (`UMBRAL_AP`, también configurable con `--umbral-ap`).

**Por qué 0,78.** Es lo que ya consigue una regresión logística sin ajustar en validación cruzada (0,775). Si el XGBoost ajustado no supera eso, algo ha cambiado en los datos o en el código y no debe reemplazar a un modelo bueno. El modelo actual da 0,813.

**Comprobado.** Con una barrera de 0,9 el script se niega a guardar y termina con código 1:

```text
$ python src/train.py --version v1 --umbral-ap 0.9
Prueba (9375 pedidos): {'average_precision': 0.8132, ...}
No se guarda el modelo: precisión media 0.8132 por debajo de la barrera 0.9.
```

La comprobación se hace antes del reentrenamiento final y antes de escribir ningún archivo; después de esa ejecución `models/` seguía vacío.

## 13 · Versionar

`--version` es obligatorio y tiene que tener la forma `v1`, `v2`… Con él se forma el nombre de los dos archivos: `models/llegada_tarde_v1.joblib` y `models/llegada_tarde_v1.json`.

**Nunca se pisa una versión.** Si ya existe cualquiera de los dos archivos, el script termina sin entrenar:

```text
$ python src/train.py --version v1
Ya existe la versión v1 en .../models: usa otra, no se sobrescribe.
```

**Qué hay en cada archivo.** El `.json` describe el `.joblib` que lo acompaña:
- versión y fecha;
- pregunta, modelo e hiperparámetros;
- columnas de entrada y derivadas, y valores conocidos de cada categoría;
- umbral de decisión y barrera usada;
- métricas de la prueba;
- número de pedidos y resumen de la limpieza;
- huella SHA-256 del CSV con el que se entrenó;
- versiones de Python, scikit-learn y XGBoost.

Con eso se sabe, sin abrir el modelo, con qué datos y con qué resultado se fabricó cada versión. Si llegan datos nuevos o se cambian hiperparámetros, se ejecuta con `--version v2` y la v1 sigue disponible para volver atrás.

## 14 · Para cerrar

Si el modelo se usara mañana en la app, miraríamos:

1. **Lo que entra.**
   - Valores nuevos de zona, restaurante o tipo de comida, que el modelo trataría como si no existieran.
   - Proporción de vacíos y de valores imposibles.
   - Que la distribución de distancia, repartidores disponibles, pedidos activos, clima y tiempo prometido se parezca a la de entrenamiento; por ejemplo, con un PSI por variable cada semana.
2. **Lo que predice.** Qué proporción de pedidos marca como «tarde» y cómo se distribuyen las probabilidades. Hoy marca alrededor de un 27 % con el umbral de 0,36. Un salto brusco sin cambio en la operación indica un problema.
3. **Lo que pasó de verdad.** `llego_tarde` se conoce poco después de cada entrega, así que se pueden calcular recall, precisión y AP de cada semana con pedidos reales, y compararlos con los de la prueba (0,74 / 0,69 / 0,81). Si bajan de forma sostenida, toca reentrenar con datos recientes como v2, y si no supera la barrera, revisar.
4. **Por segmentos.** Los mismos números por zona, vehículo y clima. Ya sabemos que la bici y la lluvia son donde más se equivoca. Un cambio en la flota o un mes muy lluvioso es lo primero que movería esos números.

## Extra: usar el modelo en una aplicación

`src/app.py` carga `models/llegada_tarde_v1.joblib` una sola vez, valida cada pedido contra el `.json` y avisa de los que probablemente lleguen tarde:

```text
$ python src/app.py
Modelo llegada_tarde_v1.joblib · umbral 0.36
Pedido 1: TARDE (probabilidad de retraso 1.00)
Pedido 2: a tiempo (probabilidad de retraso 0.00)
Pedido 3: rechazado, valor desconocido en zona: 'Marte'
```

**¿Qué pasa si le mandas una zona que no existe?**

- **Sin validación**, la tubería no da error: el one-hot tiene `handle_unknown="ignore"`, pone a cero todas las columnas de zona y predice igualmente. Con ese pedido, «Marte» da 0,024 de probabilidad, prácticamente lo mismo que «norte» (0,024) o «sur» (0,023): el modelo predice como si la zona no importara, en silencio.
- **Por eso la app valida antes de predecir**: rechaza zonas, restaurantes y demás categorías que no estaban en el entrenamiento (las lee del `.json`). También rechaza campos que faltan, números no numéricos, distancias, importes o repartidores no positivos y fechas inválidas.
