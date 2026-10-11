"""Reconstruye el dataset de la Actividad 4 (lista de las 1,638 imágenes evaluadas) sin correr los modelos.

Genera el mismo dataset.csv que la celda 8d del cuaderno: mismas imágenes, mismo orden y mismas columnas,
y una versión en Excel (dataset.xlsx) con una hoja que describe cada campo.
Usa los mismos commits fijos de Hugging Face y la misma semilla, por lo que el resultado es idéntico
en cualquier computadora. No necesita GPU ni token (los dos datasets son públicos).

Uso:
    pip install datasets huggingface_hub pandas numpy openpyxl
    python generar_dataset.py                          # genera dataset.csv y dataset.xlsx
    python generar_dataset.py --salida data/dataset.csv   # los guarda en otra carpeta (así se generó data/ del repo)
    python generar_dataset.py --descargar imagenes     # además guarda las 1,638 imágenes en esa carpeta
"""
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download

SEMILLA = 42                 # misma semilla que el cuaderno; solo interviene en la muestra de campo
POR_CLASE_CAMPO = 200

REPO_LAB, REV_LAB = "Lindll/maize-leaf-disease", "acd308e43bdec7ff3b67b0fc283eccd7f1b9423d"
MAPEO_LAB = {"Blight": "Tizón", "Common_Rust": "Roya común",
             "Gray_Leaf_Spot": "Mancha gris", "Healthy": "Sana"}

REPO_CAMPO, REV_CAMPO = "muAtarist/maize_foliar_diseases", "448bec6c7abe18fe87917e443368c63ed3786119"
MAPEO_CAMPO = {"CornLeafBlight": "Tizón", "CommonRust": "Roya común",
               "GreyLeafSpot": "Mancha gris", "NoFoliarSymptoms": "Sana"}


def dataset_laboratorio():
    """Split test completo, solo con las 4 clases de maíz (en este dataset son todas)."""
    ds = load_dataset(REPO_LAB, split="test", revision=REV_LAB)
    nombres = ds.features["label"].names
    filas = [(i, nombres[e]) for i, e in enumerate(ds["label"]) if nombres[e] in MAPEO_LAB]
    tabla = pd.DataFrame({
        "Dataset": "Laboratorio",
        "Repositorio": REPO_LAB,
        "Imagen": [f"test[{i}]" for i, _ in filas],
        "Etiqueta original": [e for _, e in filas],
        "Clase": [MAPEO_LAB[e] for _, e in filas],
    })
    return tabla, ds


def dataset_campo():
    """Muestra balanceada de 200 fotos por clase, elegida igual que en la celda 4b del cuaderno."""
    archivos = sorted(f.path for f in HfApi().list_repo_tree(REPO_CAMPO, repo_type="dataset",
                                                             revision=REV_CAMPO, recursive=True)
                      if f.path.lower().endswith((".jpg", ".jpeg", ".png")))
    por_clase = {}
    for ruta in archivos:                      # ruta = "train/<Carpeta>/<foto>.jpg"
        carpeta = ruta.split("/")[1]
        if carpeta in MAPEO_CAMPO:
            por_clase.setdefault(MAPEO_CAMPO[carpeta], []).append(ruta)
    rng = random.Random(SEMILLA)
    muestra = sorted(r for rutas in por_clase.values() for r in rng.sample(rutas, POR_CLASE_CAMPO))
    return pd.DataFrame({
        "Dataset": "Campo",
        "Repositorio": REPO_CAMPO,
        "Imagen": muestra,
        "Etiqueta original": [r.split("/")[1] for r in muestra],
        "Clase": [MAPEO_CAMPO[r.split("/")[1]] for r in muestra],
    })


DICCIONARIO = pd.DataFrame([
    ["Dataset", "texto", "Dominio de la imagen", "Laboratorio, Campo", "Campo"],
    ["Repositorio", "texto", "Repositorio de Hugging Face de donde se descarga la imagen",
     f"{REPO_LAB}, {REPO_CAMPO}", REPO_CAMPO],
    ["Imagen", "texto", "Identificador único de la imagen: en laboratorio, su índice dentro del split test; "
     "en campo, la ruta del archivo dentro del repositorio", "test[0] a test[837], train/<carpeta>/<archivo>",
     "train/GreyLeafSpot/631_20190313_121410_GLS.jpg"],
    ["Etiqueta original", "texto", "Clase tal como viene en el dataset de origen",
     ", ".join(list(MAPEO_LAB) + list(MAPEO_CAMPO)), "GreyLeafSpot"],
    ["Clase", "texto", "Clase del caso de uso a la que se tradujo la etiqueta original; es la clase real "
     "contra la que se evalúan los modelos", "Tizón, Roya común, Mancha gris, Sana", "Mancha gris"],
    ["Origen", "texto", "Tipo de imagen. En el repositorio de campo, los archivos cuyo nombre empieza con "
     "\"Corn_\" son de PlantVillage (hoja recortada, fondo uniforme); los demás son fotos tomadas en parcela",
     "Laboratorio, Tipo PlantVillage, Foto de campo", "Foto de campo"],
], columns=["Campo", "Tipo", "Descripción", "Valores posibles", "Ejemplo"])


def agregar_origen(dataset):
    """Las imágenes del repo de campo cuyo nombre empieza con "Corn_" son de PlantVillage
    (hoja recortada, fondo uniforme, 256x256); las demás son fotos tomadas en la parcela."""
    nombre = dataset["Imagen"].str.split("/").str[-1]
    dataset["Origen"] = np.where(dataset["Dataset"] == "Laboratorio", "Laboratorio",
                                 np.where(nombre.str.startswith("Corn_"), "Tipo PlantVillage", "Foto de campo"))
    return dataset


def descargar(dataset, ds_lab, carpeta):
    """Guarda cada imagen en <carpeta>/<Dataset>/<Etiqueta original>/."""
    carpeta = Path(carpeta)
    lab = dataset[dataset["Dataset"] == "Laboratorio"]
    for imagen, etiqueta in zip(lab["Imagen"], lab["Etiqueta original"]):
        i = int(imagen[5:-1])                                        # "test[123]" -> 123
        destino = carpeta / "Laboratorio" / etiqueta / f"test_{i}.png"
        destino.parent.mkdir(parents=True, exist_ok=True)
        ds_lab[i]["image"].convert("RGB").save(destino)   # algunas vienen en CMYK y PNG no lo admite
    rutas = list(dataset.loc[dataset["Dataset"] == "Campo", "Imagen"])
    snapshot_download(REPO_CAMPO, repo_type="dataset", revision=REV_CAMPO, allow_patterns=rutas,
                      local_dir=carpeta / "Campo", max_workers=16)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--salida", default="dataset.csv", help="archivo CSV a generar")
    parser.add_argument("--descargar", metavar="CARPETA", help="además descarga las imágenes a esta carpeta")
    args = parser.parse_args()

    lab, ds_lab = dataset_laboratorio()
    dataset = agregar_origen(pd.concat([lab, dataset_campo()], ignore_index=True))
    dataset.to_csv(args.salida, index=False, encoding="utf-8-sig")   # utf-8-sig para que Excel lea los acentos

    conteo = pd.crosstab([dataset["Dataset"], dataset["Origen"]], dataset["Clase"], margins=True, margins_name="Total")
    excel = Path(args.salida).with_suffix(".xlsx")
    with pd.ExcelWriter(excel) as xls:
        dataset.to_excel(xls, sheet_name="Dataset", index=False)
        DICCIONARIO.to_excel(xls, sheet_name="Diccionario", index=False)
        conteo.to_excel(xls, sheet_name="Conteo")

    print(f"{args.salida} y {excel}: {len(dataset)} imágenes")
    print(conteo)
    if args.descargar:
        descargar(dataset, ds_lab, args.descargar)
        print("Imágenes guardadas en", args.descargar)


if __name__ == "__main__":
    main()
