"""Preparación de datos, predicción y explicación que usa la aplicación web (app.py).

NOMBRES, las listas de columnas, GRUPO_RIESGO y limpiar_datos son copia exacta de las secciones 2 y 3 del notebook
Proyecto_Final_ML_Riesgo_Post_Trasplante_Enviar.ipynb; MAXIMO_HLA, nivel_hla, GRUPOS_ESCALAS y agrupar_variables, de las
secciones 5 y 6. Si cambian allá, deben cambiar aquí. cargar_artefactos comprueba que las listas coincidan con metadatos.json.

Artefactos que consume (los exporta la sección 13 del notebook a despliegue/artefactos/):
    modelo_hct.joblib       Pipeline completo de scikit-learn (preprocesamiento y modelo elegido), reentrenado con todo df_model
    metadatos.json          umbral, métricas, comparación de modelos, deciles, opciones válidas, rangos y reglas de limpieza
    fondo_shapley.csv       200 pacientes de referencia para los valores de Shapley
    pacientes_ejemplo.csv   5 pacientes del dataset (sintético) para precargar el formulario
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

CARPETA_DESPLIEGUE = Path(__file__).resolve().parent
CARPETA_ARTEFACTOS = CARPETA_DESPLIEGUE / "artefactos"
CARPETA_DATOS = CARPETA_DESPLIEGUE.parent / "data"

ARTEFACTOS = {
    "modelo": "modelo_hct.joblib",
    "metadatos": "metadatos.json",
    "fondo": "fondo_shapley.csv",
    "ejemplos": "pacientes_ejemplo.csv",
}
MIN_PACIENTES_FONDO = 50
GRUPO_HLA = "compatibilidad_hla"

NOMBRES = {
    "ID": "id", "efs": "evento", "efs_time": "meses_hasta_evento",
    "dri_score": "indice_riesgo_enfermedad", "psych_disturb": "trastorno_psiquiatrico", "cyto_score": "puntaje_citogenetico",
    "diabetes": "diabetes", "hla_match_c_high": "hla_c_alta_res", "hla_high_res_8": "hla_alta_res_8",
    "tbi_status": "irradiacion_corporal_total", "arrhythmia": "arritmia", "hla_low_res_6": "hla_baja_res_6",
    "graft_type": "tipo_injerto", "vent_hist": "antecedente_ventilacion", "renal_issue": "enfermedad_renal",
    "pulm_severe": "enfermedad_pulmonar_severa", "prim_disease_hct": "enfermedad_principal", "hla_high_res_6": "hla_alta_res_6",
    "cmv_status": "serologia_cmv", "hla_high_res_10": "hla_alta_res_10", "hla_match_dqb1_high": "hla_dqb1_alta_res",
    "tce_imm_match": "epitopos_t_inmunogenicidad", "hla_nmdp_6": "hla_nmdp_6", "hla_match_c_low": "hla_c_baja_res",
    "rituximab": "rituximab", "hla_match_drb1_low": "hla_drb1_baja_res", "hla_match_dqb1_low": "hla_dqb1_baja_res",
    "prod_type": "tipo_producto", "cyto_score_detail": "citogenetica_detalle", "conditioning_intensity": "intensidad_acondicionamiento",
    "ethnicity": "etnicidad", "year_hct": "anio_trasplante", "obesity": "obesidad", "mrd_hct": "enfermedad_minima_residual",
    "in_vivo_tcd": "deplecion_celulas_t", "tce_match": "epitopos_t_compatibilidad", "hla_match_a_high": "hla_a_alta_res",
    "hepatic_severe": "enfermedad_hepatica_severa", "donor_age": "edad_donante", "prior_tumor": "tumor_previo",
    "hla_match_b_low": "hla_b_baja_res", "peptic_ulcer": "ulcera_peptica", "age_at_hct": "edad_paciente",
    "hla_match_a_low": "hla_a_baja_res", "gvhd_proph": "profilaxis_injerto_huesped", "rheum_issue": "enfermedad_reumatologica",
    "sex_match": "sexo_donante_receptor", "hla_match_b_high": "hla_b_alta_res", "race_group": "grupo_racial",
    "comorbidity_score": "indice_comorbilidad_sorror", "karnofsky_score": "escala_karnofsky", "hepatic_mild": "enfermedad_hepatica_leve",
    "tce_div_match": "epitopos_t_diversidad", "donor_related": "donante_emparentado", "melphalan_dose": "dosis_melfalan",
    "hla_low_res_8": "hla_baja_res_8", "cardiac": "enfermedad_cardiaca", "hla_match_drb1_high": "hla_drb1_alta_res",
    "pulm_moderate": "enfermedad_pulmonar_moderada", "hla_low_res_10": "hla_baja_res_10",
}
NOMBRES_ORIGINALES = {nuevo: original for original, nuevo in NOMBRES.items()}

COLUMNAS_HLA = sorted(c for c in NOMBRES.values() if c.startswith("hla_"))
VARIABLES_GENETICAS = ["citogenetica_detalle", "epitopos_t_diversidad", "epitopos_t_inmunogenicidad", "puntaje_citogenetico"]
COLUMNAS_ENTERAS = ["edad_paciente", "edad_donante", "indice_comorbilidad_sorror", "escala_karnofsky"]
EDADES = ["edad_paciente", "edad_donante"]
COMORBILIDADES = ["trastorno_psiquiatrico", "diabetes", "arritmia", "enfermedad_renal", "enfermedad_pulmonar_severa",
                  "enfermedad_pulmonar_moderada", "enfermedad_hepatica_severa", "enfermedad_hepatica_leve", "obesidad",
                  "tumor_previo", "ulcera_peptica", "enfermedad_reumatologica", "enfermedad_cardiaca"]
GRUPO_RIESGO = {
    "Low": "Bajo", "Intermediate": "Intermedio", "Intermediate - TED AML case <missing cytogenetics": "Intermedio",
    "High": "Alto", "High - TED AML case <missing cytogenetics": "Alto", "Very high": "Muy alto",
    "N/A - non-malignant indication": "No aplica", "N/A - pediatric": "No aplica", "N/A - disease not classifiable": "No aplica",
    "TBD cytogenetics": "Pendiente", "Missing disease status": "Pendiente",
}


def limpiar_datos(df, columnas_eliminadas):
    df = df.rename(columns=NOMBRES).drop(columns=columnas_eliminadas, errors="ignore")
    # Genéticas de texto sin dato como No aplica
    for c in VARIABLES_GENETICAS:
        df[c] = df[c].astype(object).where(df[c].notna(), "No aplica").astype(str)
    # HLA como conteo entero con -1 cuando no hay dato
    for c in COLUMNAS_HLA:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(int)
    # Edades en años cumplidos y escalas como enteros
    df[["edad_paciente", "edad_donante"]] = np.floor(df[["edad_paciente", "edad_donante"]].astype(float))
    for c in COLUMNAS_ENTERAS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    # Variables nuevas
    df["dif_edad_donante"] = (df["edad_donante"] - df["edad_paciente"]).abs()
    df["es_pediatrico"] = (df["edad_paciente"] < 18).astype("Int64")
    df["n_comorbilidades"] = (df[COMORBILIDADES] == "Yes").sum(axis=1)
    irradiacion = df["irradiacion_corporal_total"]
    df["recibio_irradiacion"] = (irradiacion != "No TBI").astype("Int64").where(irradiacion.notna())
    df["grupo_riesgo_enfermedad"] = df["indice_riesgo_enfermedad"].map(GRUPO_RIESGO).fillna(df["indice_riesgo_enfermedad"])
    return df.drop(columns=["indice_riesgo_enfermedad"])


MAXIMO_HLA = {c: 2 if c.endswith("_res") else int(c.rsplit("_", 1)[1]) for c in COLUMNAS_HLA}
GRUPOS_ESCALAS = {
    "indice_comorbilidad_sorror": ([-np.inf, 0, 2, np.inf], ["0", "1 a 2", "3 o más"]),
    "escala_karnofsky": ([-np.inf, 60, 80, np.inf], ["60 o menos", "70 a 80", "90 a 100"]),
    "n_comorbilidades": ([-np.inf, 0, 1, 2, np.inf], ["0", "1", "2", "3 o más"]),
}


def nivel_hla(valores, maximo):
    fraccion = valores / maximo
    return pd.Series(np.select([valores < 0, fraccion >= 0.8, fraccion >= 0.4], ["Sin dato", "Alta", "Media"], "Baja"),
                     index=valores.index)


def agrupar_variables(df, variables):
    # Cambia cada variable indicada por su grupo con cortes fijos
    df = df.copy()
    for c in variables:
        if c in COLUMNAS_HLA:
            df[c] = nivel_hla(df[c], MAXIMO_HLA[c])
        else:
            cortes, etiquetas = GRUPOS_ESCALAS[c]
            df[c] = pd.cut(df[c].astype(float), cortes, labels=etiquetas).astype(object)
    return df


# Carga de artefactos

def artefactos_faltantes(carpeta=CARPETA_ARTEFACTOS):
    return [nombre for clave, nombre in ARTEFACTOS.items() if clave != "fondo" and not (carpeta / nombre).exists()]


def cargar_artefactos(carpeta=CARPETA_ARTEFACTOS):
    """Carga el Pipeline del notebook (columnTransformer + modelo) y lee metadatos, fondo y ejemplos.

    El fondo de Shapley es opcional: si falta o no trae las variables del formulario se devuelve None
    y la aplicación oculta la sección de explicabilidad."""
    modelo = joblib.load(carpeta / ARTEFACTOS["modelo"])
    meta = json.loads((carpeta / ARTEFACTOS["metadatos"]).read_text(encoding="utf-8"))
    for clave, local in (("variables_geneticas", VARIABLES_GENETICAS), ("columnas_hla", COLUMNAS_HLA)):
        if sorted(meta.get(clave, [])) != sorted(local):
            raise ValueError(f"{clave} de metadatos.json no coincide con procesamiento.py: vuelva a copiar la limpieza del notebook")
    ejemplos_crudos = pd.read_csv(carpeta / ARTEFACTOS["ejemplos"])
    ejemplos = formulario(ejemplos_crudos, meta)
    ejemplos.index = ejemplos_crudos["ID"].astype(str) if "ID" in ejemplos_crudos else ejemplos.index.astype(str)
    return modelo, meta, cargar_fondo(carpeta / ARTEFACTOS["fondo"], meta), ejemplos


def cargar_fondo(ruta, meta):
    if not ruta.exists():
        return None
    crudo = pd.read_csv(ruta)
    if columnas_faltantes(crudo, meta) or len(crudo) < MIN_PACIENTES_FONDO:
        return None
    return formulario(crudo, meta)


def valores_imputacion(modelo, meta):
    """Valor con el que la limpieza o el Pipeline completan cada variable del formulario cuando llega vacía."""
    valores = {c: "No aplica" for c in meta["variables_geneticas"]}
    valores.update({c: "-1 (nivel Sin dato)" if c in meta.get("variables_agrupadas", []) else "-1" for c in meta["columnas_hla"]})
    for nombre, transformador, columnas in modelo.named_steps["columnTransformer"].transformers_:
        if nombre in ("continuas", "categoricas"):
            imputador = transformador.named_steps["simpleImputer"]
        elif nombre == "binarias":
            imputador = transformador
        else:
            continue
        valores.update({c: v for c, v in zip(columnas, imputador.statistics_) if c not in valores})
    if "grupo_riesgo_enfermedad" in valores:
        valores["indice_riesgo_enfermedad"] = f"grupo {valores['grupo_riesgo_enfermedad']}"
    return valores


# Preparación de los datos

def formulario(df, meta):
    """Deja los datos del paciente con los nombres en español y solo las variables del formulario.
    Acepta columnas con los nombres originales de Kaggle o con los nombres nuevos; las que falten quedan vacías.
    Las variables numéricas quedan como enteros desde la entrada: las edades en años cumplidos (como la limpieza
    del notebook) y las escalas y conteos HLA redondeados, así lo que se muestra es lo mismo que recibe el modelo."""
    df = df.rename(columns=NOMBRES).copy()
    for c in meta["variables_formulario"]:
        if c not in df:
            df[c] = np.nan
        if c in meta["rangos"]:
            valores = pd.to_numeric(df[c], errors="coerce").astype(float)
            df[c] = np.floor(valores) if c in EDADES else valores.round()
        else:
            df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
            df[c] = df[c].map(lambda v: (v.strip() or np.nan) if isinstance(v, str) else v)
    return df[meta["variables_formulario"]].reset_index(drop=True)


def preparar_entrada(df, meta):
    """Aplica la misma limpieza y representación del notebook y entrega las columnas que recibe el modelo."""
    limpio = limpiar_datos(formulario(df, meta), meta["columnas_eliminadas"])
    return agrupar_variables(limpio[meta["columnas_entrada"]], meta.get("variables_agrupadas", []))


def valores_modelo(df, meta):
    """Valor de cada variable del formulario tal como lo recibe el modelo, después de la limpieza y la agrupación.
    Devuelve None cuando el dato falta y lo completa el Pipeline (mediana o valor más frecuente)."""
    limpio = limpiar_datos(formulario(df, meta), meta["columnas_eliminadas"])
    agrupadas = meta.get("variables_agrupadas", [])
    modelo = agrupar_variables(limpio[meta["columnas_entrada"]], agrupadas)
    salida = {}
    for c in meta["variables_formulario"]:
        if c == "indice_riesgo_enfermedad":
            v = modelo["grupo_riesgo_enfermedad"].iloc[0]
            salida[c] = None if pd.isna(v) else f"grupo {v}"
        elif c in COLUMNAS_HLA:
            numero = int(limpio[c].iloc[0])
            texto = "-1" if numero < 0 else f"{numero} de {MAXIMO_HLA[c]}"
            salida[c] = f"{texto} (nivel {modelo[c].iloc[0]})" if c in agrupadas else texto
        else:
            v = modelo[c].iloc[0]
            if pd.isna(v):
                salida[c] = None
            elif isinstance(v, (int, float, np.number)):
                salida[c] = f"{int(v)}"
            else:
                salida[c] = str(v)
    return salida


def columnas_faltantes(df, meta):
    presentes = set(df.rename(columns=NOMBRES).columns)
    return [c for c in meta["variables_formulario"] if c not in presentes]


def valores_no_reconocidos(df, meta):
    """Categorías que no aparecieron en el entrenamiento, por variable."""
    entrada = formulario(df, meta)
    salida = {c: sorted(set(entrada[c].dropna().astype(str)) - set(opciones)) for c, opciones in meta["opciones"].items()}
    return {c: v for c, v in salida.items() if v}


def valores_fuera_de_rango(df, meta):
    """Número de pacientes con valores numéricos fuera del rango visto en el entrenamiento, por variable."""
    entrada = formulario(df, meta)
    salida = {c: int(((entrada[c] < minimo) | (entrada[c] > maximo)).sum()) for c, (minimo, _, maximo) in meta["rangos"].items()}
    return {c: n for c, n in salida.items() if n}


def faltantes(df, meta):
    return formulario(df, meta).isna().sum(axis=1).to_numpy()


def a_nombres_originales(df):
    """Nombres de columnas de Kaggle (como train.csv y test.csv), para exportar archivos que se puedan volver a importar."""
    return df.rename(columns=NOMBRES_ORIGINALES)


def plantilla_csv(meta):
    return pd.DataFrame(columns=["ID"] + [NOMBRES_ORIGINALES.get(c, c) for c in meta["variables_formulario"]])


# Predicción

def predecir(modelo, df, meta):
    return modelo.predict_proba(preparar_entrada(df, meta))[:, 1].astype(float)


def nivel_riesgo(p, meta):
    """Bajo hasta el riesgo medio predicho del decil 3 de df_test, alto desde el del decil 8 (sección 10.2 del notebook)."""
    d = meta["deciles_prueba"]
    if p >= d[7]["riesgo_medio_predicho"]:
        return "Alto"
    if p <= d[2]["riesgo_medio_predicho"]:
        return "Bajo"
    return "Intermedio"


def decil(p, meta):
    """Decil de df_test cuyo riesgo medio predicho está más cerca de p."""
    d = meta["deciles_prueba"]
    return min(d, key=lambda fila: abs(fila["riesgo_medio_predicho"] - p))["decil"]


def predecir_lote(modelo, df, meta):
    """Agrega al archivo recibido el riesgo, el nivel, la alerta y el número de variables vacías de cada paciente."""
    riesgo = predecir(modelo, df, meta)
    resultado = pd.DataFrame({
        "riesgo_evento_12m": riesgo.round(4),
        "nivel": [nivel_riesgo(p, meta) for p in riesgo],
        "alto_riesgo": riesgo >= meta["umbral"],
        "variables_sin_dato": faltantes(df, meta),
    }, index=df.index)
    if "ID" in df.columns:
        resultado.insert(0, "ID", df["ID"])
    return pd.concat([resultado, df.drop(columns=[c for c in resultado.columns if c in df.columns])], axis=1)


# Explicabilidad

def grupos_explicacion(meta):
    """Cada variable del formulario es un factor, salvo las 17 HLA, que se tratan juntas como en la sección 12 del notebook."""
    hla = [c for c in meta["variables_formulario"] if c.startswith("hla_")]
    grupos = {c: [c] for c in meta["variables_formulario"] if c not in hla}
    if hla:
        grupos[GRUPO_HLA] = hla
    return grupos


def shapley_muestreo(funcion, fila, fondo, grupos, n_permutaciones=200, semilla=42):
    """Valores de Shapley por muestreo de permutaciones (el mismo concepto de SHAP, independiente del modelo).

    En cada permutación se parte de un paciente del fondo y se le van pasando, en orden aleatorio, los valores del
    paciente a explicar; lo que cambia el riesgo al pasar un factor es su contribución. Las contribuciones más el
    riesgo de referencia suman exactamente el riesgo predicho del paciente."""
    rng = np.random.default_rng(semilla)
    nombres = list(grupos)
    columnas = [c for g in nombres for c in grupos[g]]
    grupo_de_columna = np.concatenate([[k] * len(grupos[g]) for k, g in enumerate(nombres)])
    n_grupos = len(nombres)
    x = fila[columnas].to_numpy(dtype=object)[0]
    z = fondo[columnas].to_numpy(dtype=object)[rng.integers(len(fondo), size=n_permutaciones)]
    ordenes = np.array([rng.permutation(n_grupos) for _ in range(n_permutaciones)])
    posicion = np.argsort(ordenes, axis=1)
    mascara = (posicion[:, None, :] < np.arange(n_grupos + 1)[None, :, None])[:, :, grupo_de_columna]
    hibridos = pd.DataFrame(np.where(mascara, x[None, None, :], z[:, None, :]).reshape(-1, len(columnas)), columns=columnas)
    prob = funcion(hibridos).reshape(n_permutaciones, n_grupos + 1)
    contrib = np.zeros(n_grupos)
    np.add.at(contrib, ordenes.ravel(), np.diff(prob, axis=1).ravel())
    return pd.Series(contrib / n_permutaciones, index=nombres), float(prob[:, 0].mean())


def explicar(modelo, df, fondo, meta, n_permutaciones=200):
    fila = formulario(df, meta)
    return shapley_muestreo(lambda d: predecir(modelo, d, meta), fila, fondo, grupos_explicacion(meta), n_permutaciones)
