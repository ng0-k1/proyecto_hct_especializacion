# Reporte técnico: riesgo de evento en el primer año después de un trasplante de médula ósea

Proyecto final del módulo de Machine Learning. Especialización en Inteligencia Artificial, UPTC.

Autor: [Nombre]

## Resumen

| Aspecto | Resultado |
|---|---|
| Pregunta | ¿Tendrá el paciente un evento adverso en los primeros 12 meses después del trasplante? |
| Datos | 28,800 pacientes y 57 variables registradas antes del trasplante |
| Tipo de problema | Clasificación binaria, convertida desde un problema de tiempo hasta el evento |
| Modelo final | XGBClassifier ajustado con GridSearchCV |
| AUC en df_test | 0.7425 |
| Separación de riesgo | 14.0 % de pacientes con evento en el decil de menor riesgo frente a 84.1 % en el de mayor riesgo |
| Explicabilidad | SHAP sobre el modelo final, agrupado por variable original |

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
- Archivos en la carpeta `data/`: `train.csv` (28,800 pacientes, 60 columnas) y `data_dictionary.csv`
- El `test.csv` de la competencia no se usa: trae 3 pacientes sin resultado, porque Kaggle guarda las respuestas en privado
- Diccionario de variables: https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/data
- Códigos de enfermedad: https://www.manula.com/manuals/cibmtr/formsnet3-training-guide/1/en/topic/disease-codes

```python
df_raw = pd.read_csv("data/train.csv")
df_diccionario = pd.read_csv("data/data_dictionary.csv")
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

- No hay filas ni pacientes duplicados
- Dos variables superan el 50 % de nulos y se eliminan: compatibilidad de epítopos de células T (66.0 %) y enfermedad mínima residual (57.6 %)
- En las 21 variables genéticas que quedan (17 HLA, 2 de citogenética y 2 de epítopos de células T), un nulo se marca como `No aplica`: si el dato no está, lo más probable es que la prueba no se hizo o no correspondía, así que llenarlo con un promedio inventaría un resultado genético
- Se crean cinco variables nuevas

```python
def limpiar_datos(df, columnas_eliminadas, variables_geneticas):
    df = df.rename(columns=NOMBRES).drop(columns=columnas_eliminadas, errors="ignore")
    for c in variables_geneticas:
        df[c] = df[c].fillna("No aplica")                       # dato genético vacío
    df["dif_edad_donante"] = df["edad_donante"] - df["edad_paciente"]
    df["es_pediatrico"] = (df["edad_paciente"] < 18).astype(int)
    df["n_comorbilidades"] = (df[COMORBILIDADES] == "Yes").sum(axis=1)
    df["recibio_irradiacion"] = (df["irradiacion_corporal_total"] != "No TBI").astype(int)
    df["grupo_riesgo_enfermedad"] = df["indice_riesgo_enfermedad"].map(GRUPO_RIESGO)
    return df.drop(columns=["indice_riesgo_enfermedad"])
```

Todas las reglas se aplican fila por fila y no usan la variable objetivo, así que no hay fuga de información. Los nulos que quedan (28 variables, casi todas con menos de 10 %) se imputan dentro del Pipeline. La versión completa de la función está en la sección 3 del notebook y una copia exacta en `despliegue/procesamiento.py`.

## 4. De tiempo hasta el evento a clasificación

El resultado original es el tiempo en meses hasta el evento, una variable continua parecida a una regresión. El problema es la censura: a muchos pacientes se les dejó de observar sin haber tenido el evento, así que no se conoce su tiempo real. Se convierte en una pregunta de sí o no:

```text
evento_12m = 1   si el paciente tuvo el evento en los primeros 12 meses
evento_12m = 0   si tuvo el evento después, o si se le siguió más de 12 meses sin evento
excluido         si no tuvo el evento pero salió del seguimiento antes de los 12 meses
```

![Curva de Kaplan-Meier](imagenes/kaplan_meier.png)

- La decisión clínica es binaria: seguimiento intensivo o habitual
- El 96.8 % de los eventos ocurre antes de los 12 meses (mediana de 5.8 meses)
- Sin evento siguen el 70.9 % a los 6 meses, el 47.7 % a los 12 y el 46.3 % a los 18
- Solo se excluyen 238 pacientes (0.8 %); df_model queda con 28,562 pacientes, 52.6 % con evento y 47.4 % sin evento
- Partición estratificada por objetivo y grupo racial: df_train 19,992, df_val 4,285 y df_test 4,285
- El grupo racial y la etnicidad no se usan como predictores; se reservan para la revisión de equidad

## 5. Hallazgos del análisis exploratorio

![Pacientes con evento según la edad](imagenes/edad.png)

- El riesgo sube con la edad del paciente: 26 % en niños de 0 a 5 años y 66 % en mayores de 65
- La edad del donante pesa poco hasta los 53 años y sube a 57 % o 59 % con donantes mayores

![Pacientes con evento por variable clínica](imagenes/variables_clinicas.png)

- Riesgo de la enfermedad alto: 66 % con evento
- Acondicionamiento de intensidad reducida (RIC): 67 %, frente a 55 % del intenso (MAC)
- Sangre periférica: 60 %, frente a 35 % de la médula ósea
- Índice de Sorror: de 47 % con 0 puntos a 84 % con 10
- Los pacientes sin dato (barras ocre) tienen mucho menos riesgo que cualquier categoría real

### Compatibilidad HLA

![Correlación entre variables HLA](imagenes/hla_correlacion.png)

- Correlación media de 0.763 entre pares de variables HLA; las que resumen varios genes llegan a 0.99
- El 20.1 % de los pacientes no tiene `hla_alta_res_8` y su porcentaje con evento (51.7 %) es casi igual al de quienes sí lo tienen (52.9 %)

![PCA del bloque HLA](imagenes/hla_pca.png)

- Con One-Hot el bloque HLA pasa a 88 columnas; 28 componentes conservan el 95 % de la varianza

### Quién tiene datos vacíos

![Variables vacías por año](imagenes/nulos_anio.png)

![Sin dato de intensidad por enfermedad](imagenes/nulos_enfermedad.png)

- Entre 2008 y 2017 cada paciente tiene alrededor de una variable vacía; en 2020 llega a 15
- A los pacientes pediátricos les falta la intensidad del acondicionamiento en el 47.9 % de los casos, frente al 7.4 % en adultos
- En enfermedades hereditarias o no malignas (IMD, IIS, IEA, HIS) el dato falta en el 78 % o más; en AML, MDS o ALL en menos del 5 %
- No es abandono del seguimiento: los pacientes sin evento tienen entre 36 y 40 meses de seguimiento con 0 a 5 variables vacías; el grupo con 6 o más tiene 27.3 meses porque coincide con los trasplantes más recientes
- Sí es un sesgo: en los adultos, el 61.0 % de quienes tienen el dato de intensidad tuvo un evento, frente al 23.6 % de quienes no lo tienen

## 6. Preparación de datos

Todo el preprocesamiento usa clases estándar de scikit-learn dentro de un Pipeline:

```python
columnTransformer = ColumnTransformer([
    ("hla", Pipeline([("oneHotEncoder", OneHotEncoder(min_frequency=20)), ("pca", PCA(n_components=28))]), COLUMNAS_HLA),
    ("num", Pipeline([("simpleImputer", SimpleImputer(strategy="median")), ("standardScaler", StandardScaler())]), COLUMNAS_NUMERICAS),
    ("cat", Pipeline([("simpleImputer", SimpleImputer(strategy="most_frequent")), ("oneHotEncoder", OneHotEncoder(min_frequency=20))]), COLUMNAS_CATEGORICAS),
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
| XGBClassifier | constant (No informado) | 0.7412 | -31.9 puntos |
| XGBClassifier | most_frequent | 0.7378 | -6.8 puntos |

Se elige `most_frequent`: cuesta 0.003 de AUC y evita que el modelo premie los formularios incompletos.

### Reducción de dimensión

| Variante | Columnas | AUC en df_val |
|---|---|---|
| Sin PCA | 241 | 0.7377 |
| PCA solo en HLA (28 componentes) | 181 | 0.7369 |
| PCA sobre todas las columnas | 72 | 0.7315 |

### Balanceo de clases

Con 52.6 % y 47.4 % las clases están balanceadas. Se probaron pesos por clase (`compute_sample_weight`) y submuestreo de la clase mayoritaria: ninguno mejoró el AUC (mejora máxima de 0.0004) y ambos bajaron el recall del evento. No se usa balanceador; el equilibrio entre eventos detectados y alertas se ajusta con el umbral.

## 7. Comparación de modelos

StratifiedKFold de 10 pliegues sobre df_train, con el mismo Pipeline para todos:

| Modelo | AUC (media) | AUC (desv.) | Segundos por ajuste |
|---|---|---|---|
| XGBClassifier | 0.7342 | 0.0060 | 1.6 |
| LogisticRegression | 0.7314 | 0.0073 | 0.7 |
| HistGradientBoostingClassifier | 0.7279 | 0.0073 | 2.3 |
| AdaBoostClassifier | 0.7265 | 0.0078 | 10.6 |
| RandomForestClassifier | 0.7183 | 0.0101 | 4.9 |
| KNeighborsClassifier | 0.6952 | 0.0104 | 0.5 |

![Comparación de modelos](imagenes/comparacion_modelos.png)

- XGBClassifier gana en promedio y es el más estable
- Prueba de Wilcoxon contra LogisticRegression: diferencia de 0.0028, p-valor 0.0039, estadísticamente significativa

## 8. Ajuste de hiperparámetros con GridSearchCV

```python
parametrosGridSearch = {"modelo__max_depth": [3, 4, 5], "modelo__learning_rate": [0.03, 0.06],
                        "modelo__n_estimators": [200, 400], "modelo__min_child_weight": [1, 10]}
busquedaGridSearch = GridSearchCV(pipeline, parametrosGridSearch, scoring="roc_auc", cv=stratifiedKFold5)
```

![AUC de GridSearchCV](imagenes/gridsearch.png)

- 24 combinaciones, AUC entre 0.7261 y 0.7356
- Mejor combinación: profundidad 3, tasa de aprendizaje 0.06, peso mínimo por hoja 10 y 400 árboles
- En df_val: AUC 0.7429 frente a 0.7398 sin ajustar
- Umbral de alerta por índice de Youden en df_val: 0.518 (recall 0.719, precision 0.688)

## 9. Evaluación final en df_test

| Métrica | Valor |
|---|---|
| AUC-ROC | 0.7425 |
| Average precision | 0.7460 |
| F1 macro | 0.6773 |
| Balanced accuracy | 0.6770 |
| Recall del evento | 0.7168 |
| Precision del evento | 0.6872 |
| Brier score | 0.2045 |

![Matriz de confusión](imagenes/matriz_confusion.png)

- De 2,256 pacientes con evento, el modelo alertó a 1,617; de 2,029 sin evento, 736 recibieron una alerta de más

![Curva ROC](imagenes/curva_roc.png)

![Curva Precision-Recall](imagenes/curva_pr.png)

![Calibración](imagenes/calibracion.png)

- Las probabilidades están bien calibradas: los puntos siguen la diagonal de 16 % a 84 % de riesgo

![Pacientes con evento por decil de riesgo](imagenes/deciles.png)

| Nivel | Deciles | Pacientes con evento |
|---|---|---|
| Bajo | 1 a 3 | 14 % a 39 % |
| Intermedio | 4 a 7 | 42 % a 62 % |
| Alto | 8 a 10 | 70 % a 84 % |

El decil de mayor riesgo tiene 6.0 veces más eventos que el de menor riesgo.

### Equidad por grupo racial

| Grupo | Pacientes | % con evento | AUC | Recall | C-index |
|---|---|---|---|---|---|
| American Indian or Alaska Native | 710 | 49.7 | 0.739 | 0.700 | 0.661 |
| Asian | 722 | 54.2 | 0.745 | 0.762 | 0.678 |
| Black or African-American | 715 | 55.1 | 0.756 | 0.685 | 0.667 |
| More than one race | 717 | 44.5 | 0.731 | 0.614 | 0.660 |
| Native Hawaiian or other Pacific Islander | 698 | 51.1 | 0.729 | 0.703 | 0.656 |
| White | 723 | 61.1 | 0.731 | 0.803 | 0.635 |

- C-index estratificado de la competencia: 0.6464
- El recall sigue la proporción de eventos de cada grupo: donde hay más eventos, el mismo umbral detecta más

### Subgrupos clínicos

| Subgrupo | % con evento | AUC | Recall |
|---|---|---|---|
| Menor de 18 | 34.5 | 0.746 | 0.379 |
| 18 a 40 | 53.3 | 0.706 | 0.668 |
| 40 a 60 | 59.1 | 0.704 | 0.812 |
| Mayor de 60 | 64.0 | 0.700 | 0.851 |
| Médula ósea | 35.1 | 0.737 | 0.346 |
| Sangre periférica | 59.8 | 0.708 | 0.805 |

En niños el modelo ordena bien el riesgo, pero el umbral general detecta pocos eventos porque su riesgo base es menor. Conviene un umbral pediátrico propio.

## 10. Análisis a 6, 12 y 18 meses

| Horizonte | Pacientes | % excluidos por censura | % con evento | AUC |
|---|---|---|---|---|
| 6 meses | 24,496 | 0.07 | 29.1 | 0.735 |
| 12 meses | 24,277 | 0.83 | 52.6 | 0.737 |
| 18 meses | 23,589 | 3.76 | 55.6 | 0.738 |

El AUC es el mismo en los tres. A 6 meses se pierde casi la mitad de los eventos del primer año y a 18 meses se excluyen cuatro veces y media más pacientes que a 12. El horizonte de 12 meses es el mejor equilibrio y no introduce un sesgo relevante.

## 11. Explicabilidad con SHAP

`TreeExplainer` sobre 1,000 pacientes de df_val. Los valores de las 181 columnas se suman por variable original (41 variables); los componentes HLA quedan juntos como `compatibilidad_hla`. El valor base equivale a un riesgo de 52.6 %.

![Importancia SHAP](imagenes/shap_importancia.png)

| Puesto | Variable | SHAP medio absoluto |
|---|---|---|
| 1 | sexo_donante_receptor | 0.184 |
| 2 | intensidad_acondicionamiento | 0.177 |
| 3 | anio_trasplante | 0.145 |
| 4 | profilaxis_injerto_huesped | 0.145 |
| 5 | compatibilidad_hla | 0.139 |
| 6 | puntaje_citogenetico | 0.135 |
| 7 | indice_comorbilidad_sorror | 0.119 |
| 8 | deplecion_celulas_t | 0.114 |
| 9 | edad_paciente | 0.111 |
| 10 | grupo_riesgo_enfermedad | 0.108 |

![Beeswarm SHAP](imagenes/shap_beeswarm.png)

![SHAP por categoría](imagenes/shap_categorias.png)

| Variable | Sube el riesgo | Baja el riesgo |
|---|---|---|
| Sexo donante y receptor | Donante mujer, receptor hombre (F-M) | Ambos hombres (M-M) |
| Intensidad del acondicionamiento | Intensidad reducida (RIC) | Intensa (MAC) |
| Año del trasplante | Años más antiguos | Trasplantes recientes |
| Profilaxis | Tacrolimus con micofenolato (FK + MMF) | Esquemas con ciclofosfamida |
| Citogenética | Normal, mal pronóstico, intermedio o pendiente | Favorable o No aplica |
| Índice de Sorror | 3 puntos o más | 0 a 2 puntos |

![Dependencia SHAP](imagenes/shap_dependencia.png)

- Edad: los niños pequeños tienen la contribución más negativa; el efecto cruza el cero cerca de los 15 a 18 años y vuelve a subir después de los 63
- Sorror: patrón de escalera, desde 6 puntos se estabiliza en el valor más alto
- Karnofsky: sin patrón ordenado; su efecto depende de otras variables

### Explicación de pacientes individuales

![Paciente de mayor riesgo](imagenes/shap_paciente_alto.png)

- Riesgo de 94.9 %: suma por un índice de Sorror de 8, citogenética de mal pronóstico, enfermedad cardíaca, serología de citomegalovirus negativa en el donante y positiva en el receptor, enfermedad pulmonar severa y acondicionamiento reducido

![Paciente de menor riesgo](imagenes/shap_paciente_bajo.png)

- Riesgo de 6.0 %: niño de 7 años con una anomalía hereditaria de los eritrocitos (IEA), sin registro de compatibilidad HLA ni de citogenética y trasplantado en 2018
- La compatibilidad HLA en No aplica es la contribución más grande (-0.97). Es un efecto de la regla de marcar No aplica que conviene vigilar si el modelo se usa en centros donde la tipificación HLA siempre se registra

## 12. Conclusiones

- La clasificación a 12 meses responde a la decisión clínica, captura el 96.8 % de los eventos y solo excluye el 0.8 % de los pacientes
- La limpieza eliminó 2 variables con más de 50 % de nulos y marcó como No aplica los vacíos genéticos; SimpleImputer con most_frequent evitó que el modelo premiara los formularios incompletos
- XGBClassifier, ajustado con GridSearchCV, logra un AUC de 0.7425 en df_test, con probabilidades bien calibradas y un desempeño parecido en los seis grupos raciales
- El modelo separa con claridad a los pacientes: 14.0 % de eventos en el decil de menor riesgo y 84.1 % en el de mayor riesgo
- Siguientes pasos: umbral propio para pacientes pediátricos, vigilar el efecto de No aplica en HLA y recalibrar con datos locales antes de usarlo en otro centro
