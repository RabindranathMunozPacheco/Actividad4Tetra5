# Diagnóstico de enfermedades en hoja de maíz con modelos preentrenados

Comparación de tres modelos de Hugging Face (MobileNetV2, ResNet-50 y ViT-base) para la app de salud de la hoja de maíz de una asociación agrícola de Sinaloa, evaluados con fotos de laboratorio y con fotos de campo.

## 1. Descripción del proyecto

**Objetivo general:** evaluar tres modelos preentrenados de clasificación de imágenes con métricas de precisión y de eficiencia, para elegir el candidato base de una app que ayude a técnicos y productores a identificar enfermedades en la hoja de maíz a partir de una foto.

**Contexto:** la asociación reúne productores de maíz en unas 2,000 ha y les da asesoría técnica, crédito e insumos. Una enfermedad que no se detecta a tiempo reduce el rendimiento de la parcela, y un diagnóstico equivocado lleva a aplicar fungicida que no hacía falta. La asociación ya cuenta con servidor propio y sus técnicos tienen celular, así que el prototipo busca saber si un modelo preentrenado basta o si hay que adaptarlo con fotos de la región.

**Definición de la tarea:**
- *Tarea:* clasificación de imágenes multiclase (aprendizaje supervisado, visión computacional).
- *Entrada:* una foto de la hoja de maíz.
- *Salida:* una de 4 clases (Tizón, Roya común, Mancha gris o Sana) con su nivel de confianza.
- *Enfoque:* modelos preentrenados en PlantVillage (38 clases de 14 cultivos), usados solo en inferencia, sin reentrenar. Como ajuste mínimo, la salida de 38 clases se reduce a las 4 de maíz tomando la clase de maíz con mayor probabilidad.
- *Criterio de éxito:* F1 macro ≥ 0.85 en fotos de campo, con el menor número posible de hojas enfermas diagnosticadas como sanas.

**Modelos evaluados** (se fija el commit exacto con `revision` para que la descarga sea reproducible):

| Modelo | Repositorio en Hugging Face | Arquitectura | Parámetros | Tamaño | Licencia |
|---|---|---|---|---|---|
| MobileNetV2 | `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification` | CNN ligera | 2.3 M | 9.1 MB | Otra (sin términos claros) |
| ResNet-50 | `mesabo/agri-plant-disease-resnet50` | CNN residual | 23.6 M | 94.3 MB | Apache 2.0 |
| ViT-base | `kimcomehome/plantvillage-vit-leaf-disease` | Transformer de visión | 85.8 M | 343.3 MB | CC BY-SA 3.0 |

## 2. Entorno de ejecución

| Componente | Detalle |
|---|---|
| Plataforma | Google Colab |
| Lenguaje | Python 3 (la versión que trae Colab) |
| Acelerador | GPU NVIDIA T4 (16 GB). La primera ejecución se hizo con GPU; Colab la retiró antes de terminar, así que la versión guardada del cuaderno se ejecutó en CPU. Los resultados de clasificación no cambian con el dispositivo, solo la velocidad |
| Librerías | `transformers` 5.18, `datasets` 4.8, `huggingface_hub` 1.33, `torch` 2.11, `scikit-learn` 1.6, `pandas` 2.2, `matplotlib` 3.10, `Pillow` 11.3. Las versiones exactas están en `requirements.txt`, generado por la sección 2 del cuaderno |
| Autenticación | Token de Hugging Face de tipo *Read*, guardado en los *Secrets* de Colab como `HF_TOKEN` |
| Scripts de `src/` | Corren en cualquier Python 3.10 o superior, sin GPU ni token |

## 3. Estructura del repositorio

```
diagnostico-hoja-maiz/
├── README.md                     # este documento
├── requirements.txt              # librerías del cuaderno y del script
├── data/
│   ├── README.md                 # qué contiene la carpeta y cómo regenerarla
│   ├── dataset.csv               # las 1,638 imágenes evaluadas (ver sección 4)
│   └── dataset.xlsx              # lo mismo, con diccionario de campos y conteo
├── notebooks/
│   └── Act4_modelos_maiz.ipynb   # cuaderno completo, secciones 1 a 9
├── src/
│   ├── generar_dataset.py        # reconstruye data/ y, opcionalmente, descarga las imágenes
│   └── analizar_por_origen.py    # métricas en campo por origen de la foto y Excel final de resultados
└── results/
    ├── dataset_y_resultados.xlsx # dataset, predicciones y todas las métricas en un archivo
    ├── predicciones.csv          # una fila por imagen y modelo
    ├── metricas.csv              # accuracy, precision, recall y F1 por modelo y dataset
    ├── metricas_por_origen.csv   # métricas en campo separando fotos de parcela y tipo PlantVillage
    ├── latencia.csv              # tiempo por imagen en CPU (ejecución guardada del cuaderno)
    ├── latencia_gpu_t4.csv       # tiempo por imagen en GPU T4 y CPU (primera ejecución, con GPU)
    ├── resumen.csv               # tabla comparativa de precisión contra costo
    ├── f1_por_clase.csv          # F1 de cada enfermedad por modelo y dataset
    └── figuras/
        ├── ejemplos_datasets.png
        ├── matrices_confusion.png
        └── precision_vs_latencia.png
```

Las imágenes no se suben al repositorio porque se descargan de Hugging Face con un commit fijo (sección 4).

## 4. Dataset

| Dataset | Repositorio y commit | Qué contiene | Uso |
|---|---|---|---|
| Laboratorio | `Lindll/maize-leaf-disease` @ `acd308e4` | 4,188 imágenes (3,350 train / 838 test) en 4 clases. El repositorio no documenta su origen; el total y los nombres de clase coinciden con el dataset de Kaggle *Corn or Maize Leaf Disease Dataset* (PlantVillage + PlantDoc). Sin licencia declarada. | Split `test` completo: 838 imágenes |
| Campo | `muAtarist/maize_foliar_diseases` @ `448bec6c` (DOI 10.57967/hf/2691) | 7,012 JPG en 6 carpetas, una por clase. Sin ficha ni licencia. | 200 por clase (semilla 42) de las 4 carpetas que existen en los modelos: 800 imágenes |

**Hallazgo sobre el dataset de campo:** los archivos que empiezan con `Corn_` son imágenes de PlantVillage (hoja recortada, fondo uniforme, 256×256), no fotos de parcela. En la muestra de 800, **487 (61 %) son tipo PlantVillage** y **313 son fotos de campo**, y 97 de las primeras son prácticamente idénticas a imágenes del test de laboratorio. Por eso el dataset trae la columna `Origen`. Este hallazgo surgió al documentar el dataset, después de ejecutar el cuaderno, y su efecto se describe en las limitaciones (sección 7).

| Muestra de campo | Tizón | Roya común | Mancha gris | Sana | Total |
|---|---|---|---|---|---|
| Foto de campo | 78 | 58 | 140 | 37 | 313 |
| Tipo PlantVillage | 122 | 142 | 60 | 163 | 487 |

**Campos de `data/dataset.csv`** (una fila por imagen):

| Campo | Descripción | Ejemplo |
|---|---|---|
| `Dataset` | Dominio de la imagen: `Laboratorio` o `Campo` | `Campo` |
| `Repositorio` | Repositorio de Hugging Face de donde se descarga | `muAtarist/maize_foliar_diseases` |
| `Imagen` | Identificador único: índice dentro del split `test` (laboratorio) o ruta del archivo en el repositorio (campo) | `test[123]` · `train/GreyLeafSpot/631_20190313_121410_GLS.jpg` |
| `Etiqueta original` | Clase como viene en el dataset de origen | `Gray_Leaf_Spot` · `GreyLeafSpot` |
| `Clase` | Clase del caso de uso contra la que se evalúan los modelos | `Mancha gris` |
| `Origen` | `Laboratorio`, `Tipo PlantVillage` o `Foto de campo` | `Foto de campo` |

`results/predicciones.csv` agrega, por cada imagen y modelo: `Modelo`, `Real`, `Predicción` (clase de maíz con mayor score), `Score` (probabilidad de esa clase, de 0 a 1) y `Top-1 otro cultivo` (`True` si la clase más probable de las 38 era de otro cultivo).

## 5. Pasos de ejecución

**Cuaderno completo (modelos, métricas y resultados):**
1. Abrir `notebooks/Act4_modelos_maiz.ipynb` en Google Colab.
2. Activar la GPU en *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4 → Guardar*.
3. Crear en Hugging Face un token de tipo *Read* y guardarlo en los *Secrets* de Colab (ícono de llave) con el nombre `HF_TOKEN`, activando el acceso al cuaderno.
4. Ejecutar las celdas en orden:

   | Sección | Qué hace | Tiempo aproximado |
   |---|---|---|
   | 1–3 | Instala dependencias (genera `requirements.txt`) y autentica; el dispositivo, GPU o CPU, se detecta en la sección 4 | 1–2 min |
   | 4 | Carga los 3 modelos y los 2 datasets | 2–5 min |
   | 5 | Inferencia sobre las 1,638 imágenes con los 3 modelos | Pocos min con GPU; 15–20 min en CPU |
   | 6 | Métricas y matrices de confusión | Segundos |
   | 7 | Latencia por imagen en el dispositivo disponible (GPU y CPU, o solo CPU) | 1–2 min |
   | 8 | Tabla comparativa, F1 por clase y gráfica de precisión contra latencia | Segundos |
   | 9 | Conclusiones parciales (texto) | — |

5. Descargar del panel de archivos de Colab los `.csv` y `.png` que se generan, porque se borran al cerrar la sesión, y copiarlos a `results/`.

**Dataset y análisis complementario (sin modelos ni GPU):**
```bash
pip install datasets huggingface_hub pandas numpy scikit-learn openpyxl
python src/generar_dataset.py --salida data/dataset.csv                          # regenera data/ en menos de un minuto
python src/generar_dataset.py --salida data/dataset.csv --descargar data/imagenes  # además baja las 1,638 imágenes
python src/analizar_por_origen.py   # une data/dataset.csv con results/predicciones.csv y genera las métricas por origen
```
Como usa commits fijos y la semilla 42, el `dataset.csv` sale idéntico en cada ejecución. `requirements.txt` registra las versiones de la ejecución en Colab (incluye `torch` en su variante de CPU de Colab), por eso aquí se instalan solo las librerías que usan los scripts.

## 6. Resultados

**Métricas** (precision, recall y F1 en promedio macro; campo = las 800 imágenes de la muestra):

| Modelo | Dataset | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| MobileNetV2 | Laboratorio | 0.896 | 0.866 | 0.870 | 0.868 |
| ResNet-50 | Laboratorio | 0.896 | 0.869 | 0.882 | 0.873 |
| ViT-base | Laboratorio | 0.934 | 0.914 | 0.936 | 0.921 |
| MobileNetV2 | Campo | 0.725 | 0.727 | 0.725 | 0.725 |
| ResNet-50 | Campo | 0.735 | 0.753 | 0.735 | 0.738 |
| ViT-base | Campo | 0.765 | 0.804 | 0.765 | 0.771 |

**Precisión contra costo** (los errores se miden en campo):

| | MobileNetV2 | ResNet-50 | ViT-base |
|---|---|---|---|
| Caída de F1, laboratorio → campo (pts) | 14.3 | 13.5 | 15.0 |
| Top-1 de otro cultivo en campo (%) | 7.6 | 6.0 | 0.9 |
| Hoja enferma diagnosticada como sana (%) | 4.5 | 4.5 | 1.0 |
| Hoja sana diagnosticada como enferma (%) | 8.5 | 7.0 | 11.5 |
| Latencia por imagen, GPU T4 (ms) | 28.0 | 15.3 | 14.9 |
| Latencia por imagen, CPU (ms) | 55–61 | 211–234 | 383–671 |

La latencia en CPU se da como rango porque se midió en dos sesiones de Colab con resultados distintos.

Las figuras están en `results/figuras/`.

## 7. Conclusiones

1. **La brecha de dominio domina el resultado.** Los tres modelos pierden entre 13.5 y 15 puntos de F1 al pasar de fotos de laboratorio a fotos de campo, y ninguno se acerca al ~95 % que reportan sus fichas. En campo el mejor F1 macro es 0.771 (ViT), por debajo del KPI de F1 ≥ 0.85 en fotos locales que se fijó para el piloto.
2. **ViT-base es el más preciso y el que mejor equilibra los errores para el productor:** mejor F1 en ambos dominios, 0.9 % de confusión con otros cultivos y solo 1.0 % de hojas enfermas diagnosticadas como sanas (4.5 % en los otros dos). Su costo es el tamaño (343 MB), la latencia en CPU (383–671 ms según la sesión) y más falsas alarmas sobre hojas sanas (11.5 %).
3. **ResNet-50 no justifica su costo:** gana 1.3 puntos de F1 sobre MobileNetV2, dentro del margen de la muestra (±3 puntos con n = 800), con 3.5 veces más latencia en CPU y 10 veces más tamaño. La decisión real queda entre ViT y MobileNetV2.
4. **El lugar de ejecución define el modelo.** En un servidor, ViT es la mejor opción (14.9 ms en GPU T4 y menos de 1 s en CPU). MobileNetV2 (9 MB, 55–61 ms en CPU) solo se justifica si la app debe diagnosticar sin conexión en el celular, a cambio de 4.6 puntos de F1 y de más del cuádruple de hojas enfermas no detectadas.
5. **La dificultad está en distinguir enfermedades, no en detectar la hoja enferma.** La clase Sana mantiene F1 ≥ 0.89 en campo, mientras que Mancha gris (0.59–0.67) y Tizón se confunden entre sí. Roya común queda en ~0.79 con los tres modelos, sin ganancia por arquitectura.

**Siguiente paso:** usar ViT-base como candidato preliminar y hacer ajuste fino con fotos de campo etiquetadas localmente. Si la app debe correr sin conexión, repetir la comparación con MobileNetV2 ajustado y cuantizado.

**Limitaciones:**
- Solo inferencia, sin ajuste fino; los modelos no conocen enfermedades locales como la mancha de asfalto o la roya del sur.
- Dataset de campo sin ficha ni licencia y con imágenes ya reducidas a 224×224.
- La latencia se midió en Colab, no en un celular ni en el servidor real.
- Dos de los tres modelos requirieron cargar su preprocesador con una clase distinta a la que declaran.
- **Composición de la muestra de campo, detectada después de ejecutar el cuaderno.** Al documentar el dataset (sección 4) se encontró que 487 de las 800 imágenes "de campo" son de PlantVillage y solo 313 son fotos de parcela. Separándolas con `src/analizar_por_origen.py` (`results/metricas_por_origen.csv`):
  - en las imágenes tipo PlantVillage, los modelos rinden como en laboratorio (F1 0.870 / 0.873 / 0.920);
  - en las 313 fotos de parcela, el F1 baja a 0.412 / 0.423 / 0.407 (MobileNetV2 / ResNet-50 / ViT-base).

  Por lo tanto, las métricas de campo de este proyecto describen la muestra completa y sobreestiman el desempeño en fotos reales de parcela. Esto refuerza la conclusión de que ningún modelo está listo sin ajuste fino con fotos propias. ViT se mantiene como el que menos hojas enfermas diagnostica como sanas también en esas fotos (2.2 % contra 9 %).

## 8. Contacto y créditos

- **Autor:** Rabindranath Muñoz Pacheco
- **Programa:** Maestría en Inteligencia Artificial, Universidad Tecmilenio
- **Curso:** Gestión de Proyectos de Inteligencia Artificial, Actividad 4 (profesora Tania Judith Mendoza Torres)
- **Caso de estudio:** asociación agrícola de Sinaloa
- **Fecha:** octubre de 2026

Modelos y datasets de sus respectivos autores en Hugging Face (secciones 1 y 4).
