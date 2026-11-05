"""API REST del modelo de riesgo a 12 meses tras el trasplante de medula osea.

Ejecucion local:
    uvicorn api:app --reload --port 8000

Documentacion interactiva en http://localhost:8000/docs
Todos los campos son opcionales y se limpian igual que en el notebook: un dato genetico vacio se toma como
No aplica y los demas se completan con la mediana o el valor mas frecuente del entrenamiento.
"""
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import AliasChoices, ConfigDict, Field, create_model

import procesamiento as proc

modelo, meta, fondo, ejemplos = proc.cargar_artefactos()

# Cada campo acepta el nombre nuevo en español o el nombre original de Kaggle.
original = {nuevo: viejo for viejo, nuevo in meta["nombres_originales"].items()}
campos = {}
for c in meta["variables_formulario"]:
    tipo = Optional[float] if c in meta["rangos"] else Optional[str]
    alias = AliasChoices(c, original[c]) if c in original and original[c] != c else None
    campos[c] = (tipo, Field(None, description=meta["descripciones"].get(c, c), validation_alias=alias))
Paciente = create_model("Paciente", __config__=ConfigDict(populate_by_name=True), **campos)

app = FastAPI(
    title="API de riesgo post trasplante",
    description="Estima la probabilidad de un evento adverso (supervivencia libre de eventos de la competencia) en los 12 meses siguientes a un trasplante "
                "de médula ósea y explica qué variables la explican.",
    version="2.0.0",
)


def _df(pacientes):
    return pd.DataFrame([p.model_dump() for p in pacientes])


def _validar(df):
    df = proc.formulario(df, meta)
    for c, opciones in meta["opciones"].items():
        malos = set(df[c].dropna().astype(str)) - set(opciones)
        if malos:
            raise HTTPException(status_code=422, detail=f"Valor no reconocido en {c}: {sorted(malos)}. Opciones: {opciones}")


def _respuesta(p, n_sin_dato):
    return {
        "riesgo_evento_12m": round(float(p), 4),
        "alto_riesgo": bool(p >= meta["umbral"]),
        "umbral": round(meta["umbral"], 4),
        "nivel": proc.nivel_riesgo(p, meta),
        "riesgo_promedio_cohorte": round(meta["tasa_base"], 4),
        "variables_sin_dato": int(n_sin_dato),
    }


def _valor(v):
    if pd.isna(v):
        return None
    return v.item() if hasattr(v, "item") else v


@app.get("/salud")
def salud():
    return {"estado": "ok", "modelo": meta["modelo"]}


@app.get("/info")
def info():
    return {k: meta[k] for k in ("modelo", "hiperparametros", "umbral", "metricas_prueba", "c_index_estratificado",
                                 "deciles_prueba", "importancia_shap", "descripciones", "opciones", "rangos", "fuente_diccionario")}


@app.get("/ejemplo")
def ejemplo(indice: int = 0):
    if not 0 <= indice < len(ejemplos):
        raise HTTPException(status_code=404, detail=f"Hay {len(ejemplos)} ejemplos (0 a {len(ejemplos) - 1})")
    return {k: _valor(v) for k, v in ejemplos.iloc[indice].items()}


@app.post("/predecir")
def predecir(paciente: Paciente):
    df = _df([paciente])
    _validar(df)
    return _respuesta(proc.predecir(modelo, df, meta)[0], proc.faltantes(df, meta)[0])


@app.post("/explicar")
def explicar(paciente: Paciente, n_permutaciones: int = 200, top: int = 10):
    df = _df([paciente])
    _validar(df)
    riesgo = proc.predecir(modelo, df, meta)[0]
    contrib, base = proc.explicar(modelo, df, fondo, meta, n_permutaciones=min(max(n_permutaciones, 50), 800))
    orden = contrib.abs().sort_values(ascending=False).index[:top]
    entrada = proc.formulario(df, meta)
    respuesta = _respuesta(riesgo, proc.faltantes(df, meta)[0])
    respuesta.update({
        "riesgo_referencia": round(base, 4),
        "contribuciones": [{"variable": v, "descripcion": meta["descripciones"].get(v, v), "valor": _valor(entrada[v].iloc[0]),
                            "contribucion": round(float(contrib[v]), 4)} for v in orden],
    })
    return respuesta


@app.post("/predecir/lote")
def predecir_lote(pacientes: List[Paciente]):
    if not pacientes:
        raise HTTPException(status_code=422, detail="La lista de pacientes esta vacia")
    if len(pacientes) > 5000:
        raise HTTPException(status_code=413, detail="Maximo 5000 pacientes por solicitud")
    df = _df(pacientes)
    _validar(df)
    return [_respuesta(p, n) for p, n in zip(proc.predecir(modelo, df, meta), proc.faltantes(df, meta))]
