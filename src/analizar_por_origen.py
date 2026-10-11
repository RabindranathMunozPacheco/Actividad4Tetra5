"""Une las predicciones del cuaderno con el dataset y mide el desempeño en campo según el origen de cada foto.

En la muestra de campo, 487 de las 800 imágenes resultaron ser de PlantVillage (archivos "Corn_*") y solo 313
son fotos tomadas en parcela. Este script separa ambas y genera el Excel final con todos los resultados.
Hace lo mismo que las celdas 8d, 8e y 8f del cuaderno, pero a partir de los CSV ya exportados, sin correr modelos.

Uso (desde la raíz del repositorio):
    python src/analizar_por_origen.py
Lee:      data/dataset.csv, results/predicciones.csv, metricas.csv, latencia.csv, resumen.csv, f1_por_clase.csv
Genera:   results/metricas_por_origen.csv, results/predicciones.csv (con columnas Imagen y Origen)
          y results/dataset_y_resultados.xlsx
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

CLASES = ["Tizón", "Roya común", "Mancha gris", "Sana"]
MODELOS = ["MobileNetV2", "ResNet-50", "ViT-base"]
RAIZ = Path(__file__).resolve().parent.parent
DATA, RESULTS = RAIZ / "data", RAIZ / "results"


def unir(dataset, pred):
    """La inferencia recorrió cada modelo con las imágenes en el mismo orden que dataset.csv."""
    pred = pred.drop(columns=["Imagen", "Origen"], errors="ignore")
    assert len(pred) == len(dataset) * len(MODELOS)
    assert (pred["Real"].values == np.tile(dataset["Clase"].values, len(MODELOS))).all()
    for col in ["Imagen", "Origen"]:
        pred[col] = np.tile(dataset[col].values, len(MODELOS))
    return pred


def metricas_por_origen(pred):
    filas = []
    for (modelo, origen), g in pred[pred["Dataset"] == "Campo"].groupby(["Modelo", "Origen"], sort=False):
        p, r, f1, _ = precision_recall_fscore_support(g["Real"], g["Predicción"], labels=CLASES,
                                                      average="macro", zero_division=0)
        enfermas, sanas = g[g["Real"] != "Sana"], g[g["Real"] == "Sana"]
        filas.append({"Modelo": modelo, "Origen": origen, "Imágenes": len(g),
                      "Accuracy": accuracy_score(g["Real"], g["Predicción"]),
                      "Precision (macro)": p, "Recall (macro)": r, "F1 (macro)": f1,
                      "Enferma → Sana (%)": (enfermas["Predicción"] == "Sana").mean() * 100,
                      "Sana → Enferma (%)": (sanas["Predicción"] != "Sana").mean() * 100,
                      "Top-1 otro cultivo (%)": g["Top-1 otro cultivo"].mean() * 100})
    return pd.DataFrame(filas).round(3)


def f1_por_clase_origen(pred):
    filas = []
    for (modelo, origen), g in pred[pred["Dataset"] == "Campo"].groupby(["Modelo", "Origen"], sort=False):
        _, _, f1, _ = precision_recall_fscore_support(g["Real"], g["Predicción"], labels=CLASES, zero_division=0)
        filas.append({"Modelo": modelo, "Origen": origen, **dict(zip(CLASES, f1.round(3)))})
    return pd.DataFrame(filas)


def main():
    dataset = pd.read_csv(DATA / "dataset.csv", encoding="utf-8-sig")
    pred = unir(dataset, pd.read_csv(RESULTS / "predicciones.csv"))
    por_origen = metricas_por_origen(pred)
    f1_origen = f1_por_clase_origen(pred)

    pred.to_csv(RESULTS / "predicciones.csv", index=False)
    por_origen.to_csv(RESULTS / "metricas_por_origen.csv", index=False)

    with pd.ExcelWriter(RESULTS / "dataset_y_resultados.xlsx") as xls:
        dataset.to_excel(xls, sheet_name="Dataset", index=False)
        pred.to_excel(xls, sheet_name="Predicciones", index=False)
        for nombre, archivo in [("Métricas", "metricas.csv"), ("Latencia", "latencia.csv"),
                                ("Resumen", "resumen.csv"), ("F1 por clase", "f1_por_clase.csv")]:
            pd.read_csv(RESULTS / archivo).to_excel(xls, sheet_name=nombre, index=False)
        por_origen.to_excel(xls, sheet_name="Métricas por origen", index=False)
        f1_origen.to_excel(xls, sheet_name="F1 por clase y origen", index=False)

    pd.set_option("display.width", 200)
    print(por_origen.to_string(index=False))
    print()
    print(por_origen.pivot(index="Modelo", columns="Origen", values="F1 (macro)").reindex(MODELOS))
    print()
    print(f1_origen.to_string(index=False))


if __name__ == "__main__":
    main()
