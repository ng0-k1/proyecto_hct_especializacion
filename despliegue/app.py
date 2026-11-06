"""Aplicacion web del modelo de riesgo a 12 meses tras el trasplante de medula osea.

Ejecucion local:      streamlit run app.py
Streamlit Cloud:      archivo principal despliegue/app.py (ver README.md del proyecto)
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import procesamiento as proc

st.set_page_config(page_title="Riesgo post trasplante", layout="wide", initial_sidebar_state="expanded")

AZUL, ROJO, OCRE, VERDE, GRIS = "#2F5D8A", "#B5443B", "#C08A2B", "#3E7C5A", "#8C8C8C"
COLOR_NIVEL = {"Bajo": VERDE, "Intermedio": OCRE, "Alto": ROJO}
NO_INFORMADO = "Sin dato"
GRUPOS = {
    "Paciente": ["edad_paciente", "escala_karnofsky", "indice_comorbilidad_sorror", "antecedente_ventilacion", "serologia_cmv", "anio_trasplante"],
    "Enfermedad": ["enfermedad_principal", "indice_riesgo_enfermedad", "puntaje_citogenetico", "citogenetica_detalle"],
    "Donante e injerto": ["edad_donante", "donante_emparentado", "sexo_donante_receptor", "tipo_injerto", "tipo_producto"],
    "Tratamiento": ["intensidad_acondicionamiento", "irradiacion_corporal_total", "profilaxis_injerto_huesped", "deplecion_celulas_t",
                    "rituximab", "dosis_melfalan"],
    "Comorbilidades": ["trastorno_psiquiatrico", "diabetes", "arritmia", "enfermedad_renal", "enfermedad_pulmonar_severa",
                       "enfermedad_pulmonar_moderada", "enfermedad_hepatica_severa", "enfermedad_hepatica_leve", "obesidad",
                       "tumor_previo", "ulcera_peptica", "enfermedad_reumatologica", "enfermedad_cardiaca"],
    "Compatibilidad HLA (marcadores genéticos paciente y donante)": None,
    "Epítopos de células T": ["epitopos_t_inmunogenicidad", "epitopos_t_diversidad"],
}

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1220px;}
.encabezado {border-bottom: 3px solid #2F5D8A; padding-bottom: 0.6rem; margin-bottom: 1.2rem;}
.encabezado h1 {margin-bottom: 0.1rem; font-size: 2.05rem;}
.encabezado p {color: #55606B; margin: 0; font-size: 1.0rem;}
.tarjeta {background: #FFFFFF; border: 1px solid #DCE3EA; border-radius: 12px; padding: 1rem 1.2rem; height: 100%;}
.tarjeta .rotulo {font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.08em; color: #6B7682;}
.tarjeta .valor {font-size: 1.6rem; font-weight: 700; margin-top: 0.2rem;}
.tarjeta .nota {font-size: 0.85rem; color: #6B7682;}
.aviso {border-left: 5px solid var(--color); background: #FFFFFF; border-radius: 8px; padding: 0.9rem 1.1rem; margin-top: 0.6rem;}
.aviso b {color: var(--color);}
.pequeno {font-size: 0.85rem; color: #6B7682;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def cargar():
    return proc.cargar_artefactos()


@st.cache_data(show_spinner=False)
def calcular_shapley(valores, n_permutaciones):
    modelo, meta, fondo, _ = cargar()
    df = pd.DataFrame([list(valores)], columns=meta["variables_formulario"])
    return proc.explicar(modelo, df, fondo, meta, n_permutaciones)


def tarjeta(rotulo, valor, nota="", color="#1E2933"):
    st.markdown(f"<div class='tarjeta'><div class='rotulo'>{rotulo}</div><div class='valor' style='color:{color}'>{valor}</div>"
                f"<div class='nota'>{nota}</div></div>", unsafe_allow_html=True)


def lienzo(fig, alto):
    fig.update_layout(height=alto, margin=dict(l=10, r=40, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width=True)


modelo, meta, fondo, ejemplos = cargar()
def rotulo(c):
    return c.replace("_", " ").replace("hla ", "HLA ").replace("cmv", "CMV").replace("anio", "año").capitalize()

grupos = dict(GRUPOS)
grupos["Compatibilidad HLA (marcadores genéticos paciente y donante)"] = [c for c in meta["variables_formulario"] if c.startswith("hla_")]
metricas = meta["metricas_prueba"]

st.markdown("<div class='encabezado'><h1>Riesgo en el primer año después del trasplante</h1>"
            "<p>Estimación de la probabilidad de un evento adverso (supervivencia libre de eventos de la competencia) en los 12 meses siguientes a un trasplante "
            "de médula ósea, con la explicación de cada predicción.</p></div>", unsafe_allow_html=True)

# Formulario lateral
with st.sidebar:
    st.header("Datos del paciente")
    opciones_ejemplo = ["Formulario en blanco"] + [f"Paciente de ejemplo {i + 1}" for i in range(len(ejemplos))]
    eleccion = st.selectbox("Punto de partida", opciones_ejemplo, index=1,
                            help="Los ejemplos son pacientes reales del conjunto de datos de la competencia.")
    base = ejemplos.iloc[opciones_ejemplo.index(eleccion) - 1] if eleccion != "Formulario en blanco" else None
    valores = {}
    for i, (grupo, columnas) in enumerate(grupos.items()):
        with st.expander(grupo, expanded=(i == 0)):
            for c in columnas:
                defecto = None if base is None or pd.isna(base[c]) else base[c]
                clave = f"{eleccion}_{c}"
                if c in meta["rangos"]:
                    minimo, mediana, maximo = meta["rangos"][c]
                    valores[c] = st.number_input(rotulo(c), min_value=float(minimo), max_value=float(maximo),
                                                 value=None if defecto is None else float(defecto), key=clave,
                                                 placeholder=NO_INFORMADO, help=meta["descripciones"].get(c))
                else:
                    opciones = [NO_INFORMADO] + meta["opciones"][c]
                    indice = opciones.index(str(defecto)) if defecto is not None and str(defecto) in opciones else 0
                    elegido = st.selectbox(rotulo(c), opciones, index=indice, key=clave, help=meta["descripciones"].get(c))
                    valores[c] = None if elegido == NO_INFORMADO else elegido
    st.caption("Un dato genético vacío (HLA, citogenética o epítopos de células T) se toma como No aplica; los demás se completan "
               "con la mediana o el valor más frecuente del entrenamiento, igual que en el notebook.")

paciente = pd.DataFrame([{c: (np.nan if valores.get(c) is None else valores[c]) for c in meta["variables_formulario"]}])
riesgo = float(proc.predecir(modelo, paciente, meta)[0])
nivel = proc.nivel_riesgo(riesgo, meta)
alto = riesgo >= meta["umbral"]

n_sin_dato = int(proc.faltantes(paciente, meta)[0])
if n_sin_dato > 15:
    st.info(f"{n_sin_dato} de {len(meta['variables_formulario'])} variables están vacías y se completaron con valores típicos. "
            "Entre más datos reales se ingresen, más personalizada es la estimación.")

tab_riesgo, tab_exp, tab_modelo, tab_lote = st.tabs(["Riesgo", "Por qué lo predice", "Cómo funciona el modelo", "Evaluar un lote"])

with tab_riesgo:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        tarjeta("Riesgo de evento a 12 meses", f"{riesgo:.1%}", "Recaída, falla del injerto o muerte", COLOR_NIVEL[nivel])
    with c2:
        tarjeta("Nivel de riesgo", nivel, "Según los deciles de riesgo en prueba", COLOR_NIVEL[nivel])
    with c3:
        tarjeta("Clasificación del modelo", "Alto riesgo" if alto else "Riesgo no elevado",
                f"Umbral de decisión {meta['umbral']:.0%}", ROJO if alto else VERDE)
    with c4:
        tarjeta("Promedio de la cohorte", f"{meta['tasa_base']:.1%}", f"Este paciente: {riesgo / meta['tasa_base']:.2f} veces el promedio")

    st.write("")
    izq, der = st.columns([3, 2])
    with izq:
        st.subheader("Dónde queda este paciente frente a los demás")
        dec = pd.DataFrame(meta["deciles_prueba"])
        fig = go.Figure()
        fig.add_trace(go.Bar(x=dec["decil"], y=dec["tasa_real_de_evento"], name="Tasa real de evento en prueba",
                             marker_color=["#CBD5DF"] * len(dec)))
        fig.add_trace(go.Scatter(x=dec["decil"], y=dec["riesgo_medio_predicho"], name="Riesgo predicho medio", mode="lines+markers",
                                 line=dict(color="#1E2933")))
        fig.add_hline(y=riesgo, line_color=COLOR_NIVEL[nivel], line_width=3, annotation_text=f"Este paciente {riesgo:.0%}")
        fig.update_layout(yaxis=dict(tickformat=".0%", range=[0, 1]), legend=dict(orientation="h", y=-0.2),
                          xaxis_title="Decil de riesgo (D1 menor, D10 mayor)")
        lienzo(fig, 340)
    with der:
        st.subheader("Lectura para el equipo clínico")
        mensajes = {
            "Alto": "El perfil se parece al de los pacientes con mayor tasa de eventos en el primer año. Puede justificar un plan de "
                    "seguimiento más estrecho y revisar con anticipación las alternativas de donante y de tratamiento.",
            "Intermedio": "El riesgo está en la zona media de la cohorte. Conviene revisar qué factores lo elevan en la pestaña "
                          "Por qué lo predice, porque algunos pueden ser modificables antes del trasplante.",
            "Bajo": "El perfil se parece al de los pacientes con menor tasa de eventos. El seguimiento habitual parece adecuado, "
                    "sin perder de vista que el modelo no conoce todos los factores clínicos.",
        }
        st.markdown(f"<div class='aviso' style='--color:{COLOR_NIVEL[nivel]}'><b>Riesgo {nivel.lower()}</b><br>{mensajes[nivel]}</div>",
                    unsafe_allow_html=True)
        st.markdown("<p class='pequeno' style='margin-top:0.8rem'>Herramienta académica de apoyo a la decisión. "
                    "No reemplaza el criterio médico.</p>", unsafe_allow_html=True)

with tab_exp:
    st.subheader("Qué factores suben o bajan el riesgo de este paciente")
    n_perm = st.select_slider("Precisión del cálculo", options=[100, 200, 400], value=200,
                              help="Número de permutaciones del muestreo de Shapley.")
    with st.spinner("Calculando contribuciones..."):
        contrib, base_ref = calcular_shapley(tuple(paciente.iloc[0].tolist()), n_perm)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(12)
    def etiqueta(c):
        v = paciente[c].iloc[0]
        texto = NO_INFORMADO if pd.isna(v) else (f"{v:.0f}" if isinstance(v, float) else v)
        return f"{rotulo(c)}: {texto}"
    fig = go.Figure(go.Waterfall(
        orientation="h", measure=["absolute"] + ["relative"] * len(top) + ["total"],
        y=["Promedio de referencia"] + [etiqueta(c) for c in top.index] + ["Riesgo de este paciente"],
        x=[base_ref] + list(top.values) + [0],
        text=[f"{base_ref:.1%}"] + [f"{v:+.1%}" for v in top.values] + [f"{riesgo:.1%}"], textposition="outside",
        increasing=dict(marker=dict(color=ROJO)), decreasing=dict(marker=dict(color=AZUL)),
        totals=dict(marker=dict(color="#1E2933")), connector=dict(line=dict(color="#CBD5DF"))))
    fig.update_layout(xaxis=dict(tickformat=".0%"), yaxis=dict(autorange="reversed"))
    lienzo(fig, 560)
    sube = [etiqueta(c) for c in top.index if top[c] > 0.005][:3]
    baja = [etiqueta(c) for c in top.index if top[c] < -0.005][:3]
    texto = f"Partiendo de un riesgo de referencia de {base_ref:.1%}, "
    texto += ("lo que más lo aumenta es " + "; ".join(sube)) if sube else "ningún factor lo aumenta de forma clara"
    texto += (". Lo que más lo reduce es " + "; ".join(baja)) if baja else ""
    texto += f". El resultado para este paciente es {riesgo:.1%}."
    st.info(texto)
    st.caption("Valores de Shapley (el mismo concepto de SHAP) estimados por muestreo sobre las variables del registro. "
               "Se muestran los 12 factores con mayor efecto; los demás suman la diferencia restante.")

with tab_modelo:
    st.subheader("Desempeño en pacientes que el modelo nunca vio")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        tarjeta("AUC-ROC", f"{metricas['AUC-ROC']:.3f}", "Capacidad de ordenar pacientes por riesgo")
    with c2:
        tarjeta("Recall del evento", f"{metricas['recall del evento']:.1%}", "Eventos detectados con el umbral")
    with c3:
        tarjeta("Balanced accuracy", f"{metricas['balanced accuracy']:.3f}", meta["modelo"])
    st.write("")
    der,_ = st.columns(2)
    with der:
        st.markdown("Las 15 variables más importantes según SHAP")
        imp = pd.Series(meta["importancia_shap"]).sort_values(ascending=False).head(15).iloc[::-1]
        fig = go.Figure(go.Bar(x=imp.values, y=[rotulo(c) for c in imp.index], orientation="h", marker_color=AZUL))
        fig.update_layout(xaxis_title="SHAP medio absoluto (log-odds)")
        lienzo(fig, 400)
    with st.expander("Datos, método y límites"):
        st.markdown(f"""
Datos: competencia CIBMTR, Equity in post-HCT Survival Predictions, de Kaggle. Nombres de variables tomados del diccionario oficial:
{meta['fuente_diccionario']}. El modelo desplegado ({meta['modelo']}) se entrenó con {meta['pacientes_despliegue']:,} pacientes.

Método: limpieza (variables con más de 50% de nulos eliminadas y No aplica en las genéticas), variables clínicas nuevas, SimpleImputer,
escalado, One-Hot y PCA sobre el bloque HLA dentro de un Pipeline; comparación de seis modelos con validación cruzada de 10 pliegues; GridSearchCV; umbral elegido en validación; evaluación
única en un conjunto de prueba reservado; explicabilidad con SHAP.

Alcance: el registro corresponde a trasplantes de 2008 a 2020. Para usarlo en otro país o centro conviene recalibrarlo con datos locales.
""")

with tab_lote:
    st.subheader("Evaluar varios pacientes a la vez")
    st.markdown("Suba un archivo CSV con las columnas del archivo de la competencia (train.csv o test.csv) o con los nombres nuevos. "
                "Las columnas que falten se completan con valores típicos.")
    archivo = st.file_uploader("Archivo CSV", type="csv")
    if archivo is not None:
        crudo = pd.read_csv(archivo).head(30000)
        entrada = proc.formulario(crudo, meta)
        desconocidas = {c: sorted(set(entrada[c].dropna().astype(str)) - set(meta["opciones"][c])) for c in meta["opciones"]}
        desconocidas = {c: v for c, v in desconocidas.items() if v}
        if desconocidas:
            st.warning(f"Valores no vistos en el entrenamiento (se tratan como poco frecuentes): {desconocidas}")
        resultado = crudo.copy()
        resultado["riesgo_evento_12m"] = proc.predecir(modelo, crudo, meta).round(4)
        resultado["nivel"] = [proc.nivel_riesgo(p, meta) for p in resultado["riesgo_evento_12m"]]
        resultado["alto_riesgo"] = resultado["riesgo_evento_12m"] >= meta["umbral"]
        resultado["variables_sin_dato"] = proc.faltantes(crudo, meta)
        resultado = resultado.sort_values("riesgo_evento_12m", ascending=False)
        c1, c2, c3 = st.columns(3)
        with c1:
            tarjeta("Pacientes evaluados", f"{len(resultado):,}")
        with c2:
            tarjeta("Clasificados de alto riesgo", f"{resultado['alto_riesgo'].mean():.1%}", f"Umbral {meta['umbral']:.0%}", ROJO)
        with c3:
            tarjeta("Riesgo medio", f"{resultado['riesgo_evento_12m'].mean():.1%}")
        st.write("")
        fig = go.Figure(go.Histogram(x=resultado["riesgo_evento_12m"], nbinsx=40, marker_color=AZUL))
        fig.add_vline(x=meta["umbral"], line_color=ROJO, annotation_text="Umbral")
        fig.update_layout(xaxis=dict(tickformat=".0%", title="Riesgo predicho"), yaxis_title="Pacientes")
        lienzo(fig, 280)
        st.dataframe(resultado, width=True, height=360)
        st.download_button("Descargar resultados", resultado.to_csv(index=False).encode("utf-8"),
                           file_name="riesgo_post_trasplante.csv", mime="text/csv")
