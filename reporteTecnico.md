# Reporte técnico: riesgo de evento en el primer año después de un trasplante de médula ósea

Proyecto final de la materia de Aprendizaje Automático. Especialización en Inteligencia Artificial, Universidad Pedagógica y Tecnológica de Colombia.

Autor: Oscar David Perilla Dominguez (código 202624578)

Todas las cifras salen de la ejecución guardada del notebook `Proyecto_Final_ML_Riesgo_Post_Trasplante_Enviar.ipynb` y coinciden con `despliegue/artefactos/metadatos.json`. Las instrucciones de instalación y despliegue están en [README.md](README.md).

## Resumen

| Aspecto | Resultado |
|---|---|
| Pregunta | ¿Tendrá el paciente un evento adverso en los primeros 12 meses después del trasplante? |
| Datos | 28,800 pacientes y 57 variables registradas antes del trasplante |
| Tipo de problema | Clasificación binaria, convertida desde un problema de tiempo hasta el evento |
| Modelo final | XGBClassifier ajustado con GridSearchCV |
| AUC en df_test | 0.7410 |
| Separación de riesgo | 13.1 % de pacientes con evento en el decil de menor riesgo frente a 84.6 % en el de mayor riesgo |
| Explicabilidad | SHAP (TreeExplainer) en el notebook; valores de Shapley por paciente en la aplicación |
| Despliegue | Aplicación Streamlit, local o en Streamlit Community Cloud |

## 1. Problema

El trasplante alogénico de células hematopoyéticas (HCT, conocido como trasplante de médula ósea) reemplaza la médula enferma por células sanas de un donante. Puede curar leucemias y otras enfermedades graves de la sangre, pero más de la mitad de los pacientes tiene un evento adverso en el primer año.

Qué es el evento: el diccionario de la competencia llama a la variable resultado `efs`, supervivencia libre de eventos. Un paciente con `efs = 1` tuvo un evento grave después del trasplante (en trasplante este indicador agrupa sucesos como la recaída de la enfermedad, la falla del injerto o la muerte). `efs_time` dice cuántos meses pasaron hasta ese evento o hasta el último control.

El modelo estima el riesgo de cada paciente antes del trasplante para:

- Priorizar a los pacientes que necesitan un seguimiento más estrecho
- Revisar a tiempo alternativas de donante y de tratamiento
- Explicar al paciente y a su familia de qué depende su riesgo

### Cómo se mide el éxito

| Métrica | Pregunta de negocio que responde |
|---|---|
| AUC-ROC | Si tomo un paciente que tuvo el evento y otro que no, ¿con qué frecuencia el modelo le da más riesgo al primero? |
| Recall del evento | De cada 100 pacientes que tendrán un evento, ¿cuántos detecta el modelo? |
| Precision del evento | De cada 100 alertas, ¿cuántas son correctas? |
| Calibración | ¿Un riesgo de 30 % significa de verdad 3 de cada 10 pacientes? |

## 2. Datos y diccionario

- Fuente: competencia CIBMTR, Equity in post-HCT Survival Predictions, de Kaggle
- Archivos en la carpeta `data/`: `train.csv` (28,800 pacientes, 60 columnas), `data_dictionary.csv`, `test.csv` y `sample_submission.csv`
- El `test.csv` de la competencia no se usa para evaluar: trae 3 pacientes sin resultado, porque Kaggle guarda las respuestas en privado. La aplicación lo ofrece como archivo de ejemplo para la predicción por lotes
- Diccionario de variables: https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/data
- Códigos de enfermedad: https://www.manula.com/manuals/cibmtr/formsnet3-training-guide/1/en/topic/disease-codes

```python
df_raw = pd.read_csv("data/train.csv", encoding="utf-8")
df_diccionario = pd.read_csv("data/data_dictionary.csv", encoding="utf-8")
```

Ejemplos del cambio de nombres a partir del diccionario oficial:

| Nombre original | Nombre nuevo | Descripción oficial |
|---|---|---|
| `age_at_hct` | `edad_paciente` | Age at HCT |
| `comorbidity_score` | `indice_comorbilidad_sorror` | Sorror comorbidity score |
| `karnofsky_score` | `escala_karnofsky` | KPS at HCT |
| `dri_score` | `indice_riesgo_enfermedad` | Refined disease risk index |
| `conditioning_intensity` | `intensidad_acondicionamiento` | Computed planned conditioning intensity |
| `gvhd_proph` | `profilaxis_injerto_huesped` | Planned GVHD prophylaxis |
| `sex_match` | `sexo_donante_receptor` | Donor/recipient sex match |
| `hla_match_drb1_low` | `hla_drb1_baja_res` | Recipient / 1st donor antigen level (low resolution) matching at HLA-DRB1 |

Las leucemias agudas (ALL con 8,102 pacientes y AML con 7,135) y el síndrome mielodisplásico (3,046) suman más de la mitad de los pacientes.

### Variables principales en lenguaje de negocio

| Variable | Qué significa |
|---|---|
| Escala de Karnofsky | Estado funcional del paciente, de 0 a 100. 100 es una vida normal sin síntomas |
| Índice de comorbilidad de Sorror | Suma ponderada de otras enfermedades del paciente. Más puntos, más fragilidad |
| Índice de riesgo de la enfermedad | Qué tan agresiva es la enfermedad que motiva el trasplante |
| Intensidad del acondicionamiento | Qué tan fuerte es la quimioterapia o radiación previa. MAC es intensa, RIC de intensidad reducida y NMA no mieloablativa |
| Profilaxis injerto contra huésped | Medicamentos para evitar que las células del donante ataquen al paciente |
| Compatibilidad HLA | Parecido entre los marcadores genéticos del sistema inmune del paciente y del donante. Más compatibilidad, menos rechazo |
| Citogenética | Alteraciones en los cromosomas de las células enfermas, que indican qué tan agresiva es la enfermedad |
| Sexo donante y receptor | Combinación de sexos, por ejemplo F-M es donante mujer y receptor hombre |

## 3. Calidad y limpieza: de df_raw a df_clean

![Valores nulos por variable](imagenes/nulos_variables.png)

- No hay filas ni pacientes duplicados; hay 35 variables categóricas y 25 numéricas
- Dos variables superan el 50 % de nulos y se eliminan: compatibilidad de epítopos de células T (66.0 %) y enfermedad mínima residual (57.6 %)
- En las 21 variables genéticas que quedan (17 HLA, 2 de citogenética y 2 de epítopos de células T), un nulo se marca como `No aplica`, entre 8.3 % y 41.4 % de los pacientes según la variable: si el dato no está, lo más probable es que la prueba no se hizo o no correspondía, así que llenarlo con un promedio inventaría un resultado genético
- Se crean cinco variables nuevas: diferencia de edad donante-paciente, paciente pediátrico, número de comorbilidades, si recibió irradiación corporal total y el índice de riesgo de la enfermedad agrupado
- df_clean queda con 62 columnas (60 originales, menos 2 eliminadas, más 5 nuevas, menos el índice de riesgo original)

```python
def limpiar_datos(df, columnas_eliminadas, variables_geneticas):
    df = df.rename(columns=NOMBRES).drop(columns=columnas_eliminadas, errors="ignore")
    for c in variables_geneticas:
        if c in df.columns:
            texto = df[c].map(lambda v: f"{v:g}" if isinstance(v, (int, float)) and not pd.isna(v) else v)
            df[c] = texto.fillna("No aplica").astype(str)
    df["dif_edad_donante"] = df["edad_donante"] - df["edad_paciente"]
    df["es_pediatrico"] = (df["edad_paciente"] < 18).astype(int)
    df["n_comorbilidades"] = (df[COMORBILIDADES] == "Yes").sum(axis=1)
    df["recibio_irradiacion"] = (df["irradiacion_corporal_total"] != "No TBI").astype(int)
    df["grupo_riesgo_enfermedad"] = df["indice_riesgo_enfermedad"].map(GRUPO_RIESGO).fillna(df["indice_riesgo_enfermedad"])
    return df.drop(columns=["indice_riesgo_enfermedad"])
```

Todas las reglas se aplican fila por fila y no usan la variable objetivo, así que no hay fuga de información. Los nulos que quedan (28 variables, casi todas con menos de 10 %) se imputan dentro del Pipeline. La función está en la sección 3 del notebook y una copia exacta en `despliegue/procesamiento.py`, que es la que usa la aplicación.

## 4. De tiempo hasta el evento a clasificación

El resultado original es el tiempo en meses hasta el evento, una variable continua parecida a una regresión. El problema es la censura: a muchos pacientes se les dejó de observar sin haber tenido el evento, así que no se conoce su tiempo real. Se convierte en una pregunta de sí o no:

```text
evento_12m = 1   si el paciente tuvo el evento en los primeros 12 meses
evento_12m = 0   si tuvo el evento después, o si se le siguió 12 meses o más sin evento
excluido         si no tuvo el evento pero salió del seguimiento antes de los 12 meses
```

![Curva de Kaplan-Meier](imagenes/kaplan_meier.png)

- La decisión clínica es binaria: seguimiento intensivo o habitual
- De los 15,532 pacientes con evento, el 96.8 % lo tuvo antes de los 12 meses (mediana de 5.8 meses)
- Sin evento siguen el 70.9 % a los 6 meses, el 47.7 % a los 12 y el 46.3 % a los 18
- Solo se excluyen 238 pacientes (0.8 %); df_model queda con 28,562 pacientes y 57 predictores, 52.6 % con evento y 47.4 % sin evento
- Partición 70/15/15 estratificada por objetivo y grupo racial: df_train 19,992, df_val 4,285 y df_test 4,285, los tres con 52.6 % de eventos
- El grupo racial y la etnicidad no se usan como predictores; solo sirven para estratificar la partición

## 5. Hallazgos del análisis exploratorio

Todo el análisis usa solo df_train.

![Pacientes con evento según la edad](imagenes/edad.png)

- El riesgo sube con la edad del paciente: 26 % en niños de 0 a 5 años, 52 % (el promedio) entre 30 y 36 años y 66 % en mayores de 65
- La edad del donante pesa poco hasta los 53 años y sube a 57 % o 59 % con donantes mayores

![Pacientes con evento por variable clínica](imagenes/variables_clinicas.png)

- Riesgo de la enfermedad alto: 66 % con evento; la categoría Muy alto queda por debajo del promedio (41 %)
- Acondicionamiento de intensidad reducida (RIC): 67 %, frente a 55 % del intenso (MAC)
- Sangre periférica: 60 %, frente a 35 % de la médula ósea
- Índice de Sorror: de 47 % con 0 puntos a 84 % con 10
- Escala de Karnofsky: no sigue un orden perfecto (50 tiene menos eventos que 60 u 80)
- Los pacientes sin dato (barras ocre) tienen menos riesgo que el promedio; sin dato de intensidad solo el 20 % tiene evento

### Compatibilidad HLA

![Correlación entre variables HLA](imagenes/hla_correlacion.png)

- Correlación media de 0.763 entre pares de variables HLA (mínima 0.494); las que resumen varios genes llegan a 0.99
- El 20.1 % de los pacientes no tiene `hla_alta_res_8` y su porcentaje con evento (51.7 %) es casi igual al de quienes sí lo tienen (52.9 %)

![PCA del bloque HLA](imagenes/hla_pca.png)

- Con One-Hot el bloque HLA pasa a 88 columnas; el primer componente explica el 41.6 % y 28 componentes conservan el 95 % de la varianza

### Quién tiene datos vacíos

![Variables vacías por año](imagenes/nulos_anio.png)

![Sin dato de intensidad por enfermedad](imagenes/nulos_enfermedad.png)

- Entre 2008 y 2017 cada paciente tiene alrededor de una variable vacía de las 27 que aún pueden tener nulos; en 2020 llega a 15
- A los pacientes pediátricos les falta la intensidad del acondicionamiento en el 47.9 % de los casos, frente al 7.4 % en adultos
- En enfermedades hereditarias o no malignas (IMD, IIS, IEA, HIS) el dato falta en el 78 % o más, y en el 82 % de los pocos casos de linfoma de Hodgkin; en AML, MDS o ALL en menos del 5 %
- No es abandono del seguimiento: los pacientes sin evento tienen entre 35.9 y 39.5 meses de seguimiento en mediana con 0 a 5 variables vacías; el grupo con 6 o más tiene 27.3 meses porque coincide con los trasplantes más recientes
- Sí es un sesgo: en los adultos, el 61.0 % de quienes tienen el dato de intensidad tuvo un evento, frente al 23.6 % de quienes no lo tienen; en los niños, 48.4 % frente a 18.9 %

## 6. Preparación de datos

Todo el preprocesamiento usa clases estándar de scikit-learn dentro de un Pipeline, así que en la validación cruzada cada paso aprende solo de los pliegues de entrenamiento:

```python
columnTransformer = ColumnTransformer([
    ("hla", Pipeline([("oneHotEncoder", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False)),
                      ("pca", PCA(n_components=28, random_state=42))]), COLUMNAS_HLA),
    ("num", Pipeline([("simpleImputer", SimpleImputer(strategy="median")), ("standardScaler", StandardScaler())]), COLUMNAS_NUMERICAS),
    ("cat", Pipeline([("simpleImputer", SimpleImputer(strategy="most_frequent")),
                      ("oneHotEncoder", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False))]),
     COLUMNAS_CATEGORICAS),
])
pipelineFinal = Pipeline([("columnTransformer", columnTransformer), ("modelo", XGBClassifier(...))])
```

El modelo recibe 57 predictores (17 HLA, 9 numéricos y 31 categóricos) que se convierten en 181 columnas.

### Imputación

Cambio de riesgo con vacíos: diferencia en el riesgo promedio al borrar la intensidad del acondicionamiento, la escala de Karnofsky y el índice de Sorror a 3,447 pacientes completos de df_val.

| Modelo | SimpleImputer | AUC en df_val | Cambio de riesgo con vacíos |
|---|---|---|---|
| LogisticRegression | constant (No informado) | 0.7397 | -21.8 puntos |
| LogisticRegression | most_frequent | 0.7369 | -5.4 puntos |
| XGBClassifier | constant (No informado) | 0.7407 | -31.4 puntos |
| XGBClassifier | most_frequent | 0.7378 | -6.7 puntos |

Se elige `most_frequent`: cuesta 0.003 de AUC y evita que el modelo premie los formularios incompletos.

### Reducción de dimensión

Con LogisticRegression en df_val:

| Variante | Columnas | AUC en df_val |
|---|---|---|
| Sin PCA | 241 | 0.7377 |
| PCA solo en HLA (28 componentes) | 181 | 0.7369 |
| PCA sobre todas las columnas (95 % de varianza) | 72 | 0.7315 |

Se usa PCA solo en el bloque HLA: quita 60 columnas con una pérdida mínima de AUC.

### Balanceo de clases

Con 52.6 % y 47.4 % las clases están balanceadas. Se probaron pesos por clase (`compute_sample_weight`) y submuestreo de la clase mayoritaria: ninguno mejoró el AUC (mejora máxima de 0.0004) y ambos bajaron el recall del evento de XGBClassifier de 0.750 a cerca de 0.70. No se usa balanceador; el equilibrio entre eventos detectados y alertas se ajusta con el umbral.

## 7. Comparación de modelos

StratifiedKFold de 10 pliegues sobre df_train, con el mismo Pipeline para todos (KNN con un PCA global adicional):

| Modelo | AUC (media) | AUC (desv.) | Average precision | Balanced accuracy | Segundos por ajuste |
|---|---|---|---|---|---|
| XGBClassifier | 0.7345 | 0.0059 | 0.7391 | 0.6711 | 3.4 |
| LogisticRegression | 0.7314 | 0.0073 | 0.7373 | 0.6660 | 2.2 |
| HistGradientBoostingClassifier | 0.7275 | 0.0070 | 0.7328 | 0.6649 | 5.3 |
| AdaBoostClassifier | 0.7265 | 0.0078 | 0.7348 | 0.6616 | 17.2 |
| RandomForestClassifier | 0.7183 | 0.0101 | 0.7236 | 0.6559 | 14.8 |
| KNeighborsClassifier | 0.6952 | 0.0104 | 0.6965 | 0.6429 | 1.1 |

![Comparación de modelos](imagenes/comparacion_modelos.png)

- XGBClassifier gana en promedio y es el más estable entre pliegues
- LogisticRegression queda muy cerca y su caja se cruza con la de XGBClassifier; KNN es claramente el más débil

## 8. Ajuste de hiperparámetros con GridSearchCV

```python
parametrosGridSearch = {"modelo__max_depth": [3, 4, 5], "modelo__learning_rate": [0.03, 0.06],
                        "modelo__n_estimators": [200, 400], "modelo__min_child_weight": [1, 10]}
busquedaGridSearch = GridSearchCV(pipeline, parametrosGridSearch, scoring="roc_auc", cv=stratifiedKFold5)
```

![AUC de GridSearchCV](imagenes/gridsearch.png)

- 24 combinaciones, AUC entre 0.7260 y 0.7360: el modelo es poco sensible a los hiperparámetros
- Mejor combinación: profundidad 3, tasa de aprendizaje 0.06, peso mínimo por hoja 1 y 400 árboles
- AUC de entrenamiento del mejor candidato: 0.8011, un sobreajuste moderado que los árboles poco profundos ayudan a contener
- En df_val: AUC 0.7428 frente a 0.7410 sin ajustar
- Umbral de alerta por índice de Youden en df_val: 0.504 (recall 0.742, precision 0.680), fijado antes de ver df_test

## 9. Evaluación final en df_test

df_test (4,285 pacientes) se usa una sola vez, con el modelo y el umbral ya decididos.

| Métrica | Valor |
|---|---|
| AUC-ROC | 0.7410 |
| Average precision | 0.7445 |
| F1 macro | 0.6736 |
| Balanced accuracy | 0.6734 |
| Recall del evento | 0.7332 |
| Precision del evento | 0.6784 |
| Brier score | 0.2051 |

![Matriz de confusión](imagenes/matriz_confusion.png)

- De 2,256 pacientes con evento, el modelo alertó a 1,654 y no detectó a 602; de 2,029 sin evento, 1,245 quedaron bien clasificados y 784 recibieron una alerta de más

![Curva ROC](imagenes/curva_roc.png)

![Curva Precision-Recall](imagenes/curva_pr.png)

- Sin modelo, alertar a todos daría una precision de 52.6 %; el modelo mantiene 80 % o más mientras detecta cerca del 30 % de los eventos

![Calibración](imagenes/calibracion.png)

- Las probabilidades están bien calibradas: los puntos siguen la diagonal de 16 % a 84 % de riesgo

![Pacientes con evento por decil de riesgo](imagenes/deciles.png)

| Nivel | Deciles | Riesgo predicho medio | Pacientes con evento |
|---|---|---|---|
| Bajo | 1 a 3 | 16 % a 38 % | 13.1 % a 39.2 % |
| Intermedio | 4 a 7 | 45 % a 63 % | 42.5 % a 62.6 % |
| Alto | 8 a 10 | 68 % a 84 % | 70.4 % a 84.6 % |

El decil de mayor riesgo tiene 6.5 veces más eventos que el de menor riesgo. La tasa real sube decil a decil con una sola excepción leve (D6 con 54.4 % frente a D5 con 55.0 %). La aplicación usa estos cortes: nivel bajo hasta el riesgo medio del decil 3 (37.6 %) y alto desde el del decil 8 (67.8 %).

### Subgrupos clínicos

| Subgrupo | Pacientes | % con evento | AUC | Recall |
|---|---|---|---|---|
| Menor de 18 | 1,042 | 34.5 | 0.745 | 0.398 |
| 18 a 40 | 1,051 | 53.3 | 0.704 | 0.686 |
| 40 a 60 | 1,332 | 59.1 | 0.703 | 0.825 |
| Mayor de 60 | 860 | 64.0 | 0.699 | 0.869 |
| Médula ósea | 1,236 | 35.1 | 0.732 | 0.362 |
| Sangre periférica | 3,049 | 59.8 | 0.707 | 0.822 |

En niños el modelo ordena bien el riesgo, pero el umbral general detecta pocos eventos porque su riesgo base es menor; lo mismo pasa con el injerto de médula ósea. Conviene un umbral pediátrico propio.

## 10. Análisis a 6, 12 y 18 meses

El mismo Pipeline, con StratifiedKFold de 5 pliegues sobre los pacientes que no están en df_test:

| Horizonte | Pacientes | % excluidos por censura | % con evento | AUC |
|---|---|---|---|---|
| 6 meses | 24,496 | 0.07 | 29.1 | 0.734 |
| 12 meses | 24,277 | 0.83 | 52.6 | 0.737 |
| 18 meses | 23,589 | 3.76 | 55.6 | 0.738 |

El AUC es prácticamente el mismo en los tres. A 6 meses se pierde casi la mitad de los eventos del primer año y a 18 meses se excluyen cuatro veces y media más pacientes que a 12. El horizonte de 12 meses es el mejor equilibrio y no introduce un sesgo relevante.

## 11. Explicabilidad con SHAP

`TreeExplainer` sobre 2,000 pacientes de df_val con el modelo final. Los valores de las 181 columnas se suman por variable original (41 variables); los 28 componentes HLA quedan juntos como `compatibilidad_hla`. El valor base es 0.105 en log-odds, que equivale a un riesgo de 52.6 %.

![Importancia SHAP](imagenes/shap_importancia.png)

| Puesto | Variable | SHAP medio absoluto |
|---|---|---|
| 1 | sexo_donante_receptor | 0.184 |
| 2 | intensidad_acondicionamiento | 0.179 |
| 3 | profilaxis_injerto_huesped | 0.140 |
| 4 | compatibilidad_hla | 0.138 |
| 5 | anio_trasplante | 0.137 |
| 6 | indice_comorbilidad_sorror | 0.126 |
| 7 | puntaje_citogenetico | 0.122 |
| 8 | deplecion_celulas_t | 0.116 |
| 9 | grupo_riesgo_enfermedad | 0.108 |
| 10 | serologia_cmv | 0.107 |

![Beeswarm SHAP](imagenes/shap_beeswarm.png)

![SHAP por categoría](imagenes/shap_categorias.png)

| Variable | Sube el riesgo | Baja el riesgo |
|---|---|---|
| Sexo donante y receptor | Donante mujer, receptor hombre (F-M), seguido de M-F | Ambos hombres (M-M) |
| Intensidad del acondicionamiento | Intensidad reducida (RIC) | Intensa (MAC) |
| Profilaxis | Tacrolimus con micofenolato (FK + MMF) | Esquemas con ciclofosfamida |
| Año del trasplante | Años más antiguos | Trasplantes recientes |
| Citogenética | Normal, mal pronóstico, otro, intermedio o pendiente | Favorable y, aún más, No aplica |
| Índice de Sorror | 3 puntos o más | 0 a 2 puntos |

![Dependencia SHAP](imagenes/shap_dependencia.png)

- Edad: los niños pequeños tienen la contribución más negativa (alrededor de -0.4 a -0.5); el efecto cruza el cero cerca de los 15 años, se mantiene estable en los adultos y vuelve a subir cerca de los 65
- Sorror: patrón de escalera; de 0 a 2 puntos baja el riesgo, desde 3 lo sube y desde 6 se estabiliza en el valor más alto
- Karnofsky: sin patrón ordenado; su efecto depende de otras variables

### Explicación de pacientes individuales

![Paciente de mayor riesgo](imagenes/shap_paciente_alto.png)

- Riesgo de 95.9 %: lo empujan un índice de Sorror de 8, citogenética de mal pronóstico, enfermedad cardíaca, serología de citomegalovirus negativa en el donante y positiva en el receptor (-/+), acondicionamiento reducido, enfermedad pulmonar severa, grupo de riesgo Alto y 6 comorbilidades; la combinación M-M es lo único relevante que lo baja

![Paciente de menor riesgo](imagenes/shap_paciente_bajo.png)

- Riesgo de 6.0 %: niño de 7 años con una anomalía hereditaria de los eritrocitos (IEA), sin registro de compatibilidad HLA ni de citogenética, trasplantado en 2018 y con depleción de células T
- La compatibilidad HLA en No aplica es la contribución más grande (-0.97). Es un efecto de la regla de marcar No aplica que conviene vigilar si el modelo se usa en centros donde la tipificación HLA siempre se registra

## 12. Exportación y despliegue

La sección 13 del notebook reentrena el Pipeline final con los 28,562 pacientes de df_model y los hiperparámetros elegidos, y exporta a `artefactos/`:

| Archivo | Contenido |
|---|---|
| `preprocesamiento_hct.joblib` | ColumnTransformer ajustado (imputación, escalado, One-Hot y PCA del bloque HLA) |
| `modelo_hct.json` | XGBClassifier ajustado, en el formato nativo de xgboost |
| `metadatos.json` | Umbral, métricas y deciles de df_test, importancia SHAP, reglas de limpieza, opciones válidas y rangos del formulario |
| `fondo_shapley.csv` | 200 pacientes de train.csv (sin `efs` ni `efs_time`) como referencia para las explicaciones |
| `pacientes_ejemplo.csv` | 5 pacientes de train.csv para probar la aplicación |
| `requirements_modelo.txt` | scikit-learn 1.6.1, pandas 2.2.3, numpy 2.4.6, joblib 1.6.0 y xgboost 3.4.1 |

Las métricas, el umbral y los deciles se calcularon con el modelo antes del reentrenamiento; el modelo desplegado usa todos los pacientes para aprovechar más datos.

La aplicación (`despliegue/app.py`, con la lógica en `despliegue/procesamiento.py`) reconstruye el Pipeline a partir del preprocesamiento y del estimador, y aplica la misma `limpiar_datos` del notebook antes de predecir:

- Un paciente: se crea en un formulario (en blanco o a partir de un ejemplo) o se importa desde un CSV, se predice con el botón **Predecir riesgo** y se exporta a CSV con el resultado
- Por lotes: se importa un CSV con los nombres de columnas de Kaggle o en español (o se usa `data/test.csv`), se predice con **Predecir lote** y se exportan los resultados
- Explicabilidad: valores de Shapley del modelo desplegado para cada paciente, estimados por muestreo de permutaciones con los 200 pacientes de `fondo_shapley.csv` como referencia; las 17 variables HLA se tratan como un solo factor, como `compatibilidad_hla`. Están en puntos de probabilidad, y sumados al riesgo de referencia dan exactamente el riesgo del paciente. La sección solo aparece si el fondo existe y trae las variables del formulario

Las dependencias están en un único `requirements.txt`, con las versiones del modelo fijadas, y la aplicación se publica gratis en Streamlit Community Cloud (pasos en el README).

## 13. Conclusiones

- La clasificación a 12 meses responde a la decisión clínica, captura el 96.8 % de los eventos y solo excluye el 0.8 % de los pacientes; los horizontes de 6 y 18 meses dan el mismo AUC pero pierden eventos o pacientes
- La limpieza eliminó 2 variables con más de 50 % de nulos y marcó como No aplica los vacíos de las 21 variables genéticas; SimpleImputer con most_frequent evitó que el modelo premiara los formularios incompletos (de 31.4 a 6.7 puntos de cambio en el riesgo)
- PCA redujo el bloque HLA de 88 a 28 columnas sin perder capacidad; las clases están balanceadas y ningún balanceador mejoró el AUC
- XGBClassifier fue el mejor de seis modelos (AUC 0.7345 en validación cruzada) y, ajustado con GridSearchCV, logra en df_test un AUC de 0.7410, recall de 73.3 % y precision de 67.8 %, con probabilidades bien calibradas
- El modelo separa con claridad a los pacientes: 13.1 % de eventos en el decil de menor riesgo y 84.6 % en el de mayor riesgo. Ordena bien el riesgo en todos los grupos de edad (AUC de 0.699 a 0.745), aunque con el umbral general detecta pocos eventos en los niños
- SHAP muestra que el riesgo depende del donante (combinación de sexos, compatibilidad HLA), del tratamiento (intensidad del acondicionamiento, profilaxis), del año del trasplante y de la fragilidad del paciente (Sorror, edad)
- Siguientes pasos: un umbral propio para pacientes pediátricos, una revisión de equidad del desempeño por grupo racial, vigilar el efecto de No aplica en la compatibilidad HLA y recalibrar el modelo con datos locales antes de usarlo en otro centro
