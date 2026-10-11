# data

| Archivo | Contenido |
|---|---|
| `dataset.csv` | Las 1,638 imágenes evaluadas (838 de laboratorio y 800 de campo), una por fila |
| `dataset.xlsx` | Lo mismo en Excel, con las hojas *Diccionario* (descripción de cada campo) y *Conteo* |

Las imágenes no se suben al repositorio: se descargan de Hugging Face con un commit fijo. Para obtenerlas, o para volver a generar estos archivos:

```bash
python src/generar_dataset.py --salida data/dataset.csv                       # regenera los dos archivos
python src/generar_dataset.py --salida data/dataset.csv --descargar data/imagenes   # además baja las imágenes
```

El origen de cada dataset y la descripción de los campos están en la sección 4 del README principal.
