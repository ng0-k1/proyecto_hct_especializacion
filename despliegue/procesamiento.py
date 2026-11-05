"""Lógica compartida por la API (api.py) y la aplicación web (app.py).

NOMBRES, COMORBILIDADES, GRUPO_RIESGO y limpiar_datos son copia exacta de las secciones 2 y 3 del notebook.
El modelo exportado es un Pipeline de scikit-learn con clases estándar, así que se carga directo con joblib.
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

CARPETA_ARTEFACTOS = Path(__file__).parent / "artefactos"

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

COMORBILIDADES = ["trastorno_psiquiatrico", "diabetes", "arritmia", "enfermedad_renal", "enfermedad_pulmonar_severa",
                  "enfermedad_pulmonar_moderada", "enfermedad_hepatica_severa", "enfermedad_hepatica_leve", "obesidad",
                  "tumor_previo", "ulcera_peptica", "enfermedad_reumatologica", "enfermedad_cardiaca"]
GRUPO_RIESGO = {
    "Low": "Bajo", "Intermediate": "Intermedio", "Intermediate - TED AML case <missing cytogenetics": "Intermedio",
    "High": "Alto", "High - TED AML case <missing cytogenetics": "Alto", "Very high": "Muy alto",
    "N/A - non-malignant indication": "No aplica", "N/A - pediatric": "No aplica", "N/A - disease not classifiable": "No aplica",
    "TBD cytogenetics": "Pendiente", "Missing disease status": "Pendiente",
}


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


def cargar_artefactos(carpeta=CARPETA_ARTEFACTOS):
    modelo = joblib.load(carpeta / "modelo_hct.joblib")
    meta = json.loads((carpeta / "metadatos.json").read_text(encoding="utf-8"))
    fondo = formulario(pd.read_csv(carpeta / "fondo_shapley.csv"), meta)
    ejemplos = formulario(pd.read_csv(carpeta / "pacientes_ejemplo.csv"), meta)
    return modelo, meta, fondo, ejemplos


def formulario(df, meta):
    """Deja los datos del paciente como en df_raw, con los nombres en español y las variables del formulario.
    Acepta columnas con los nombres originales de Kaggle o con los nombres nuevos."""
    df = df.rename(columns=NOMBRES).copy()
    for c in meta["variables_formulario"]:
        if c not in df:
            df[c] = np.nan
        if c in meta["rangos"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        else:
            df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
    return df[meta["variables_formulario"]].reset_index(drop=True)


def preparar_entrada(df, meta):
    """Aplica la misma limpieza del notebook y entrega las columnas que recibe el modelo."""
    limpio = limpiar_datos(formulario(df, meta), meta["columnas_eliminadas"], meta["variables_geneticas"])
    return limpio[meta["columnas_entrada"]]


def predecir(modelo, df, meta):
    return modelo.predict_proba(preparar_entrada(df, meta))[:, 1]


def faltantes(df, meta):
    return formulario(df, meta).isna().sum(axis=1).to_numpy()


def nivel_riesgo(p, meta):
    """Bajo hasta el riesgo medio del decil 3 de prueba, alto desde el decil 8."""
    d = meta["deciles_prueba"]
    if p >= d[7]["riesgo_medio_predicho"]:
        return "Alto"
    if p <= d[2]["riesgo_medio_predicho"]:
        return "Bajo"
    return "Intermedio"


def shapley_muestreo(funcion, fila, fondo, columnas, n_permutaciones=200, semilla=42):
    """Valores de Shapley por muestreo de permutaciones sobre las variables del formulario (mismo concepto que SHAP)."""
    rng = np.random.default_rng(semilla)
    p = len(columnas)
    x = fila[columnas].to_numpy(dtype=object)[0]
    z = fondo[columnas].to_numpy(dtype=object)[rng.integers(len(fondo), size=n_permutaciones)]
    ordenes = np.array([rng.permutation(p) for _ in range(n_permutaciones)])
    mascara = np.argsort(ordenes, axis=1)[:, None, :] < np.arange(p + 1)[None, :, None]
    hibridos = pd.DataFrame(np.where(mascara, x[None, None, :], z[:, None, :]).reshape(-1, p), columns=columnas)
    prob = funcion(hibridos).reshape(n_permutaciones, p + 1)
    contrib = np.zeros(p)
    np.add.at(contrib, ordenes.ravel(), np.diff(prob, axis=1).ravel())
    return pd.Series(contrib / n_permutaciones, index=columnas), float(prob[:, 0].mean())


def explicar(modelo, df, fondo, meta, n_permutaciones=200):
    fila = formulario(df, meta)
    return shapley_muestreo(lambda d: predecir(modelo, d, meta), fila, fondo, meta["variables_formulario"], n_permutaciones)
