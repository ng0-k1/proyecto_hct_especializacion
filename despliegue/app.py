"""Aplicación web del modelo de riesgo a 12 meses tras el trasplante de médula ósea.

Ejecución local, desde la raíz del proyecto:   streamlit run despliegue/app.py
Streamlit Community Cloud:                     Main file path = despliegue/app.py (ver README.md)
"""
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import procesamiento as proc

st.set_page_config(page_title="Riesgo post trasplante", layout="wide", initial_sidebar_state="expanded")

AZUL, ROJO, OCRE, VERDE, TINTA, CLARO = "#2F5D8A", "#B5443B", "#C08A2B", "#3E7C5A", "#1E2933", "#CBD5DF"
COLOR_NIVEL = {"Bajo": VERDE, "Intermedio": OCRE, "Alto": ROJO}
SIN_DATO = "Sin dato"
MAX_FILAS_LOTE = 30_000

GRUPOS = {
    "Paciente": ["edad_paciente", "escala_karnofsky", "indice_comorbilidad_sorror", "antecedente_ventilacion", "serologia_cmv",
                 "anio_trasplante"],
    "Enfermedad": ["enfermedad_principal", "indice_riesgo_enfermedad", "puntaje_citogenetico", "citogenetica_detalle"],
    "Donante e injerto": ["edad_donante", "donante_emparentado", "sexo_donante_receptor", "tipo_injerto", "tipo_producto"],
    "Tratamiento": ["intensidad_acondicionamiento", "irradiacion_corporal_total", "profilaxis_injerto_huesped", "deplecion_celulas_t",
                    "rituximab", "dosis_melfalan"],
    "Comorbilidades": ["trastorno_psiquiatrico", "diabetes", "arritmia", "enfermedad_renal", "enfermedad_pulmonar_severa",
                       "enfermedad_pulmonar_moderada", "enfermedad_hepatica_severa", "enfermedad_hepatica_leve", "obesidad",
                       "tumor_previo", "ulcera_peptica", "enfermedad_reumatologica", "enfermedad_cardiaca"],
    "Compatibilidad HLA (paciente y donante)": [],
    "Epítopos de células T": ["epitopos_t_inmunogenicidad", "epitopos_t_diversidad"],
}
ETIQUETAS = {
    "edad_paciente": "Edad del paciente (años)", "escala_karnofsky": "Escala de Karnofsky",
    "indice_comorbilidad_sorror": "Índice de comorbilidad de Sorror", "antecedente_ventilacion": "Antecedente de ventilación mecánica",
    "serologia_cmv": "Serología CMV (donante/receptor)", "anio_trasplante": "Año del trasplante",
    "enfermedad_principal": "Enfermedad principal", "indice_riesgo_enfermedad": "Índice de riesgo de la enfermedad (DRI)",
    "puntaje_citogenetico": "Puntaje citogenético", "citogenetica_detalle": "Citogenética para DRI (AML/MDS)",
    "edad_donante": "Edad del donante (años)", "donante_emparentado": "Donante emparentado",
    "sexo_donante_receptor": "Sexo donante-receptor", "tipo_injerto": "Tipo de injerto", "tipo_producto": "Tipo de producto",
    "intensidad_acondicionamiento": "Intensidad del acondicionamiento", "irradiacion_corporal_total": "Irradiación corporal total (TBI)",
    "profilaxis_injerto_huesped": "Profilaxis injerto contra huésped", "deplecion_celulas_t": "Depleción de células T in vivo",
    "rituximab": "Rituximab en el acondicionamiento", "dosis_melfalan": "Melfalán", "trastorno_psiquiatrico": "Trastorno psiquiátrico",
    "diabetes": "Diabetes", "arritmia": "Arritmia", "enfermedad_renal": "Enfermedad renal moderada o severa",
    "enfermedad_pulmonar_severa": "Enfermedad pulmonar severa", "enfermedad_pulmonar_moderada": "Enfermedad pulmonar moderada",
    "enfermedad_hepatica_severa": "Enfermedad hepática moderada o severa", "enfermedad_hepatica_leve": "Enfermedad hepática leve",
    "obesidad": "Obesidad", "tumor_previo": "Tumor sólido previo", "ulcera_peptica": "Úlcera péptica",
    "enfermedad_reumatologica": "Enfermedad reumatológica", "enfermedad_cardiaca": "Enfermedad cardíaca",
    "epitopos_t_inmunogenicidad": "Epítopos T: inmunogenicidad", "epitopos_t_diversidad": "Epítopos T: diversidad",
    "compatibilidad_hla": "Compatibilidad HLA", "grupo_riesgo_enfermedad": "Grupo de riesgo de la enfermedad",
    "n_comorbilidades": "Número de comorbilidades", "es_pediatrico": "Paciente pediátrico",
    "dif_edad_donante": "Diferencia de edad donante-paciente", "recibio_irradiacion": "Recibió irradiación",
}
PASOS = {"edad_paciente": (1.0, "%.1f"), "edad_donante": (1.0, "%.1f"), "escala_karnofsky": (10.0, "%.0f")}
LECTURA = {
    "Alto": "El perfil se parece al de los pacientes con mayor tasa de eventos en el primer año. Puede justificar un plan de "
            "seguimiento más estrecho y revisar con anticipación las alternativas de donante y de tratamiento.",
    "Intermedio": "El riesgo está en la zona media de la cohorte. Conviene revisar qué factores lo elevan en la pestaña "
                  "Por qué lo predice, porque algunos pueden ser modificables antes del trasplante.",
    "Bajo": "El perfil se parece al de los pacientes con menor tasa de eventos. El seguimiento habitual parece adecuado, "
            "sin perder de vista que el modelo no conoce todos los factores clínicos.",
}

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1220px;}
.encabezado {border-bottom: 3px solid #2F5D8A; padding-bottom: 0.6rem; margin-bottom: 1.2rem;}
.encabezado h1 {margin-bottom: 0.1rem; font-size: 2.0rem;}
.encabezado p {color: #55606B; margin: 0; font-size: 1.0rem;}
.tarjeta {background: #FFFFFF; border: 1px solid #DCE3EA; border-radius: 12px; padding: 1rem 1.2rem; height: 100%;}
.tarjeta .rotulo {font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.08em; color: #6B7682;}
.tarjeta .valor {font-size: 1.55rem; font-weight: 700; margin-top: 0.2rem;}
.tarjeta .nota {font-size: 0.85rem; color: #6B7682;}
.aviso {border-left: 5px solid var(--color); background: #FFFFFF; border-radius: 8px; padding: 0.9rem 1.1rem; margin-top: 0.6rem;}
.aviso b {color: var(--color);}
.pequeno {font-size: 0.85rem; color: #6B7682;}
</style>
""", unsafe_allow_html=True)


# Artefactos

faltan = proc.artefactos_faltantes()
if faltan:
    st.error(f"Faltan artefactos del modelo en despliegue/artefactos/: {', '.join(faltan)}. "
             "Se generan con la sección 13 del notebook.")
    st.stop()


@st.cache_resource(show_spinner="Cargando el modelo...")
def cargar():
    modelo, meta, fondo, ejemplos = proc.cargar_artefactos()
    return modelo, meta, fondo, ejemplos, proc.valores_imputacion(modelo, meta)


modelo, meta, fondo, ejemplos, imputacion = cargar()
HAY_EXPLICACION = fondo is not None
VARIABLES = meta["variables_formulario"]
grupos = {g: [c for c in cols if c in VARIABLES] for g, cols in GRUPOS.items()}
grupos["Compatibilidad HLA (paciente y donante)"] = [c for c in VARIABLES if c.startswith("hla_")]
otras = [c for c in VARIABLES if not any(c in cols for cols in grupos.values())]
if otras:
    grupos["Otras variables"] = otras


# Utilidades

def etiqueta(c):
    if c in ETIQUETAS:
        return ETIQUETAS[c]
    if c.startswith("hla_"):
        partes = c.split("_")
        resolucion = "alta resolución" if "alta" in partes else "baja resolución"
        if partes[1] == "nmdp":
            return "HLA NMDP: A y B baja, DRB1 alta (de 6)"
        if partes[-1].isdigit():
            return f"HLA total, {resolucion} (de {partes[-1]})"
        return f"HLA-{partes[1].upper()}, {resolucion} (de 2)"
    return c.replace("_", " ").capitalize()


def texto_valor(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return SIN_DATO
    if isinstance(v, (float, np.floating)):
        return f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}"
    return str(v)


def texto_imputacion(c):
    v = imputacion.get(c)
    if v is None:
        return "valor típico"
    if isinstance(v, (int, float, np.number)):
        return f"mediana {float(v):.4g}"
    if v == "No aplica" or c == "indice_riesgo_enfermedad":
        return v
    return f"más frecuente: {v}"


def tarjeta(rotulo, valor, nota="", color=TINTA):
    st.markdown(f"<div class='tarjeta'><div class='rotulo'>{rotulo}</div><div class='valor' style='color:{color}'>{valor}</div>"
                f"<div class='nota'>{nota}</div></div>", unsafe_allow_html=True)


def lienzo(fig, alto):
    fig.update_layout(height=alto, margin=dict(l=10, r=40, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def a_csv(df):
    return df.to_csv(index=False).encode("utf-8")


@st.cache_data(show_spinner=False)
def leer_csv(contenido):
    """Lee un CSV separado por comas o por punto y coma (como lo exporta Excel en español)."""
    primera = contenido.split(b"\n", 1)[0]
    separador = ";" if primera.count(b";") > primera.count(b",") else ","
    return pd.read_csv(io.BytesIO(contenido), sep=separador, decimal="," if separador == ";" else ".", low_memory=False)


def leer_archivo(archivo):
    try:
        datos = leer_csv(archivo.getvalue())
    except Exception as error:
        st.error(f"No se pudo leer el archivo: {error}")
        return None
    if datos.empty:
        st.warning("El archivo no tiene filas.")
        return None
    return datos


@st.cache_data(show_spinner="Calculando contribuciones...")
def calcular_shapley(paciente, n_permutaciones):
    return proc.explicar(modelo, paciente, fondo, meta, n_permutaciones)


def ajustar_base(base):
    """Ajusta un paciente importado al formulario: recorta valores fuera de rango y deja vacías las categorías desconocidas."""
    if base is None:
        return None, []
    base, avisos = base.copy(), []
    for c in VARIABLES:
        v = base[c]
        if pd.isna(v):
            continue
        if c in meta["rangos"]:
            minimo, _, maximo = meta["rangos"][c]
            if not minimo <= v <= maximo:
                base[c] = min(max(v, minimo), maximo)
                avisos.append(f"{etiqueta(c)}: {v:g} fuera del rango del entrenamiento, se ajustó a {base[c]:g}")
        elif str(v) not in meta["opciones"][c]:
            base[c] = np.nan
            avisos.append(f"{etiqueta(c)}: '{v}' no es una opción conocida, se dejó como {SIN_DATO}")
    return base, avisos


def campo(c, base, clave):
    defecto = None if base is None or pd.isna(base[c]) else base[c]
    descripcion = meta["descripciones"].get(c, "")
    vacio = f"{SIN_DATO} ({texto_imputacion(c)})"
    if c in meta["rangos"]:
        minimo, _, maximo = meta["rangos"][c]
        paso, formato = PASOS.get(c, (1.0, "%.0f"))
        return st.number_input(etiqueta(c), min_value=float(minimo), max_value=float(maximo),
                               value=None if defecto is None else float(defecto), step=paso, format=formato,
                               key=f"{clave}__{c}", placeholder=vacio,
                               help=f"{descripcion}. Rango del entrenamiento: {minimo:g} a {maximo:g}.")
    opciones = [None] + meta["opciones"][c]
    indice = opciones.index(str(defecto)) if defecto is not None and str(defecto) in opciones else 0
    return st.selectbox(etiqueta(c), opciones, index=indice, key=f"{clave}__{c}", help=descripcion,
                        format_func=lambda v: vacio if v is None else v)


def grafico_deciles(riesgos=None, marca=None, color_marca=TINTA):
    dec = pd.DataFrame(meta["deciles_prueba"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dec["decil"], y=dec["tasa_real_de_evento"], name="% real con evento en df_test", marker_color=CLARO))
    fig.add_trace(go.Scatter(x=dec["decil"], y=dec["riesgo_medio_predicho"], name="Riesgo predicho medio", mode="lines+markers",
                             line=dict(color=TINTA)))
    if marca is not None:
        fig.add_hline(y=marca, line_color=color_marca, line_width=3, annotation_text=f"Este paciente {marca:.0%}",
                      annotation_position="top left")
    fig.update_layout(yaxis=dict(tickformat=".0%", range=[0, 1]), legend=dict(orientation="h", y=-0.22),
                      xaxis_title="Decil de riesgo en df_test (D1 menor, D10 mayor)")
    return fig


# Encabezado

st.markdown("<div class='encabezado'><h1>Riesgo en el primer año después del trasplante</h1>"
            "<p>Probabilidad de un evento adverso (recaída, falla del injerto o muerte; variable efs de la competencia CIBMTR) "
            "en los 12 meses siguientes a un trasplante de médula ósea, con la explicación de cada predicción.</p></div>",
            unsafe_allow_html=True)


# Panel lateral: crear, cargar o importar los datos de un paciente

with st.sidebar:
    st.header("Datos del paciente")
    origen = st.radio("Punto de partida", ["Paciente de ejemplo", "Formulario en blanco", "Importar CSV"], key="origen",
                      help="Puede partir de un paciente real de ejemplo, llenar el formulario desde cero o importar un CSV "
                           "(por ejemplo, uno exportado antes desde esta aplicación).")
    base, clave, avisos, id_paciente = None, "blanco", [], None
    if origen == "Paciente de ejemplo":
        id_paciente = st.selectbox("Paciente", list(ejemplos.index), format_func=lambda i: f"ID {i}",
                                   help="Pacientes reales de train.csv, guardados por el notebook en pacientes_ejemplo.csv.")
        base, clave = ejemplos.loc[id_paciente], f"ejemplo_{id_paciente}"
    elif origen == "Importar CSV":
        archivo = st.file_uploader("CSV con uno o varios pacientes", type="csv", key="csv_paciente",
                                   help="Columnas de train.csv o test.csv de Kaggle, o los nombres en español. "
                                        "Las columnas que falten quedan como Sin dato.")
        clave = "csv_sin_archivo"
        if archivo is not None:
            crudo = leer_archivo(archivo)
            if crudo is not None:
                ids = crudo["ID"].astype(str) if "ID" in crudo else pd.Series([str(i + 1) for i in range(len(crudo))])
                fila = 0
                if len(crudo) > 1:
                    fila = st.selectbox(f"Paciente del archivo ({len(crudo):,} filas)", range(len(crudo)),
                                        format_func=lambda i: f"Fila {i + 1} · ID {ids.iloc[i]}")
                base, avisos = ajustar_base(proc.formulario(crudo.iloc[[fila]], meta).iloc[0])
                id_paciente, clave = ids.iloc[fila], f"csv_{archivo.name}_{archivo.size}_{fila}"
                faltan_cols = proc.columnas_faltantes(crudo, meta)
                if faltan_cols:
                    avisos.insert(0, f"{len(faltan_cols)} variables no vienen en el archivo y quedan como {SIN_DATO}")
                for aviso in avisos:
                    st.caption(f"⚠ {aviso}")

    with st.form("form_paciente", border=False):
        valores = {}
        for i, (grupo, columnas) in enumerate(grupos.items()):
            with st.expander(grupo, expanded=(i == 0)):
                for c in columnas:
                    valores[c] = campo(c, base, clave)
        predecir_individual = st.form_submit_button("Predecir riesgo", type="primary", width="stretch")
    st.caption("Un dato vacío se completa igual que en el notebook: las variables genéticas (HLA, citogenética y epítopos de "
               "células T) quedan como No aplica y las demás toman la mediana o el valor más frecuente del entrenamiento.")

if predecir_individual:
    paciente = proc.formulario(pd.DataFrame([{c: (np.nan if valores[c] is None else valores[c]) for c in VARIABLES}]), meta)
    st.session_state["individual"] = {"clave": clave, "paciente": paciente, "id": id_paciente}

estado = st.session_state.get("individual")
vigente = estado is not None and estado["clave"] == clave
if vigente:
    paciente = estado["paciente"]
    riesgo = float(proc.predecir(modelo, paciente, meta)[0])
    nivel = proc.nivel_riesgo(riesgo, meta)
    alto = riesgo >= meta["umbral"]
    n_sin_dato = int(proc.faltantes(paciente, meta)[0])

nombres_tabs = ["Paciente", "Por qué lo predice", "Predicción por lotes", "Cómo funciona el modelo"]
if not HAY_EXPLICACION:
    nombres_tabs.remove("Por qué lo predice")
tabs = dict(zip(nombres_tabs, st.tabs(nombres_tabs)))


def pedir_prediccion():
    st.info("Elija el punto de partida en el panel lateral (paciente de ejemplo, formulario en blanco o CSV importado), "
            "revise o complete los datos y pulse **Predecir riesgo**.")


# Pestaña: paciente individual

with tabs["Paciente"]:
    if not vigente:
        pedir_prediccion()
        c1, c2, c3 = st.columns(3)
        with c1:
            tarjeta("1. Crear", "Formulario", "Llene los datos desde cero o parta de uno de los 5 pacientes de ejemplo.")
        with c2:
            tarjeta("2. Importar", "CSV", "Cargue un paciente con las columnas de train.csv o test.csv, o un archivo exportado aquí.")
        with c3:
            tarjeta("3. Exportar", "CSV", "Descargue los datos del paciente con su riesgo para guardarlos o volver a importarlos.")
    else:
        if n_sin_dato > 15:
            st.warning(f"{n_sin_dato} de {len(VARIABLES)} variables están vacías y se completaron con valores típicos. "
                       "Entre más datos reales se ingresen, más personalizada es la estimación.")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            tarjeta("Riesgo de evento a 12 meses", f"{riesgo:.1%}", "Recaída, falla del injerto o muerte", COLOR_NIVEL[nivel])
        with c2:
            tarjeta("Nivel de riesgo", nivel, f"Comparable al decil {proc.decil(riesgo, meta)} de df_test", COLOR_NIVEL[nivel])
        with c3:
            tarjeta("Alerta del modelo", "Alto riesgo" if alto else "Sin alerta",
                    f"Umbral de decisión {meta['umbral']:.1%}", ROJO if alto else VERDE)
        with c4:
            tarjeta("Promedio de la cohorte", f"{meta['tasa_base']:.1%}",
                    f"Este paciente: {riesgo / meta['tasa_base']:.2f} veces el promedio")

        st.write("")
        izq, der = st.columns([3, 2])
        with izq:
            st.subheader("Dónde queda frente a los demás pacientes")
            lienzo(grafico_deciles(marca=riesgo, color_marca=COLOR_NIVEL[nivel]), 340)
        with der:
            st.subheader("Lectura para el equipo clínico")
            st.markdown(f"<div class='aviso' style='--color:{COLOR_NIVEL[nivel]}'><b>Riesgo {nivel.lower()}</b><br>{LECTURA[nivel]}</div>",
                        unsafe_allow_html=True)
            st.markdown("<p class='pequeno' style='margin-top:0.8rem'>Niveles: bajo hasta el riesgo medio del decil 3 "
                        f"({meta['deciles_prueba'][2]['riesgo_medio_predicho']:.0%}), alto desde el del decil 8 "
                        f"({meta['deciles_prueba'][7]['riesgo_medio_predicho']:.0%}). Herramienta académica de apoyo a la "
                        "decisión; no reemplaza el criterio médico.</p>", unsafe_allow_html=True)

        st.subheader("Exportar")
        exportado = proc.a_nombres_originales(paciente)
        exportado.insert(0, "ID", estado["id"] if estado["id"] is not None else "")
        exportado = exportado.assign(riesgo_evento_12m=round(riesgo, 4), nivel=nivel, alto_riesgo=alto, variables_sin_dato=n_sin_dato)
        e1, e2 = st.columns(2)
        with e1:
            st.download_button("Descargar paciente y resultado (CSV)", a_csv(exportado), on_click="ignore",
                               file_name=f"paciente_{estado['id'] or 'nuevo'}_riesgo.csv", mime="text/csv", width="stretch",
                               help="Usa los nombres de columnas de Kaggle: se puede volver a importar aquí o en Predicción por lotes.")
        with e2:
            st.download_button("Descargar plantilla vacía (CSV)", a_csv(proc.plantilla_csv(meta)), on_click="ignore",
                               file_name="plantilla_pacientes.csv", mime="text/csv", width="stretch", key="plantilla_individual")

        with st.expander(f"Datos usados en la predicción ({len(VARIABLES) - n_sin_dato} ingresados, {n_sin_dato} completados)"):
            fila = paciente.iloc[0]
            tabla = pd.DataFrame({
                "Grupo": [g for g, cols in grupos.items() for _ in cols],
                "Variable": [etiqueta(c) for cols in grupos.values() for c in cols],
                "Valor ingresado": [texto_valor(fila[c]) for cols in grupos.values() for c in cols],
                "Valor que usa el modelo": [texto_valor(fila[c]) if not pd.isna(fila[c]) else texto_imputacion(c)
                                            for cols in grupos.values() for c in cols],
            })
            st.dataframe(tabla, hide_index=True, height=420)


# Pestaña: explicabilidad (solo si fondo_shapley.csv es utilizable)

if HAY_EXPLICACION:
    with tabs["Por qué lo predice"]:
        st.subheader("Qué factores suben o bajan el riesgo de este paciente")
        if not vigente:
            pedir_prediccion()
        else:
            o1, o2 = st.columns(2)
            with o1:
                n_perm = st.select_slider("Precisión del cálculo (permutaciones)", options=[100, 200, 400], value=200,
                                          help="Más permutaciones dan contribuciones más estables y tardan un poco más.")
            with o2:
                n_top = st.slider("Factores a mostrar", 5, 20, 12)
            contrib, base_ref = calcular_shapley(paciente, n_perm)
            orden = contrib.abs().sort_values(ascending=False).index
            top, resto = contrib[orden[:n_top]], contrib[orden[n_top:]]

            def rotulo_factor(c):
                if c == proc.GRUPO_HLA:
                    v = paciente["hla_alta_res_8"].iloc[0]
                    return f"{etiqueta(c)}: {'No aplica' if pd.isna(v) else f'{v:.0f} de 8'}"
                return f"{etiqueta(c)}: {texto_valor(paciente[c].iloc[0])}"

            etiquetas = ["Riesgo de referencia"] + [rotulo_factor(c) for c in top.index]
            valores_x, medidas = [base_ref] + list(top.values), ["absolute"] + ["relative"] * len(top)
            textos = [f"{base_ref:.1%}"] + [f"{v:+.1%}" for v in top.values]
            if len(resto):
                etiquetas.append(f"Otros {len(resto)} factores")
                valores_x.append(float(resto.sum()))
                medidas.append("relative")
                textos.append(f"{resto.sum():+.1%}")
            fig = go.Figure(go.Waterfall(
                orientation="h", measure=medidas + ["total"], y=etiquetas + ["Riesgo de este paciente"], x=valores_x + [0],
                text=textos + [f"{riesgo:.1%}"], textposition="outside",
                increasing=dict(marker=dict(color=ROJO)), decreasing=dict(marker=dict(color=AZUL)),
                totals=dict(marker=dict(color=TINTA)), connector=dict(line=dict(color=CLARO))))
            fig.update_layout(xaxis=dict(tickformat=".0%"), yaxis=dict(autorange="reversed"))
            lienzo(fig, 140 + 32 * len(etiquetas))

            sube = [rotulo_factor(c) for c in top.index if top[c] > 0.005][:3]
            baja = [rotulo_factor(c) for c in top.index if top[c] < -0.005][:3]
            texto = f"Partiendo de un riesgo de referencia de {base_ref:.1%}, "
            texto += ("lo que más lo aumenta es " + "; ".join(sube)) if sube else "ningún factor lo aumenta de forma clara"
            texto += (". Lo que más lo reduce es " + "; ".join(baja)) if baja else ""
            texto += f". El resultado para este paciente es {riesgo:.1%}."
            st.info(texto)

            with st.expander("Todas las contribuciones"):
                st.dataframe(pd.DataFrame({"Factor": [rotulo_factor(c) for c in orden],
                                           "Contribución (puntos de riesgo)": (contrib[orden] * 100).round(2).values}),
                             hide_index=True, height=400)
            st.caption(f"Valores de Shapley del modelo desplegado, estimados por muestreo de {n_perm} permutaciones con los "
                       f"{len(fondo)} pacientes de referencia de fondo_shapley.csv. Las 17 variables HLA se tratan como un solo "
                       "factor, igual que compatibilidad_hla en la sección 12 del notebook. Las contribuciones están en puntos "
                       "de probabilidad y, sumadas al riesgo de referencia, dan exactamente el riesgo del paciente; el notebook "
                       "las reporta en log-odds, así que se compara la dirección y el orden, no la magnitud.")


# Pestaña: predicción por lotes

with tabs["Predicción por lotes"]:
    st.subheader("Predecir varios pacientes a la vez")
    st.markdown("Importe un CSV con las columnas de `train.csv` o `test.csv` de Kaggle, o con los nombres en español. "
                "Las columnas que falten quedan como Sin dato y las columnas extra (por ejemplo `efs` o `race_group`) se "
                "conservan en el resultado, pero no se usan para predecir.")
    fuentes = {"Subir un archivo CSV": None}
    if (proc.CARPETA_DATOS / "test.csv").exists():
        fuentes["data/test.csv (3 pacientes de Kaggle)"] = proc.CARPETA_DATOS / "test.csv"
    fuentes["pacientes_ejemplo.csv (5 pacientes)"] = proc.CARPETA_ARTEFACTOS / proc.ARTEFACTOS["ejemplos"]
    f1, f2 = st.columns([3, 1])
    with f1:
        fuente = st.radio("Origen de los datos", list(fuentes), horizontal=True, key="fuente_lote")
    with f2:
        st.download_button("Plantilla vacía (CSV)", a_csv(proc.plantilla_csv(meta)), file_name="plantilla_pacientes.csv",
                           mime="text/csv", on_click="ignore", width="stretch", key="plantilla_lote")

    crudo_lote, clave_lote = None, None
    if fuentes[fuente] is None:
        archivo_lote = st.file_uploader("Archivo CSV", type="csv", key="csv_lote")
        if archivo_lote is not None:
            crudo_lote, clave_lote = leer_archivo(archivo_lote), f"{archivo_lote.name}_{archivo_lote.size}"
    else:
        crudo_lote, clave_lote = pd.read_csv(fuentes[fuente]), fuente

    if crudo_lote is not None:
        if len(crudo_lote) > MAX_FILAS_LOTE:
            st.warning(f"El archivo tiene {len(crudo_lote):,} filas; se evalúan las primeras {MAX_FILAS_LOTE:,}.")
            crudo_lote = crudo_lote.head(MAX_FILAS_LOTE)
        faltan_cols = proc.columnas_faltantes(crudo_lote, meta)
        desconocidos = proc.valores_no_reconocidos(crudo_lote, meta)
        fuera = proc.valores_fuera_de_rango(crudo_lote, meta)
        reconocidas = len(VARIABLES) - len(faltan_cols)
        st.caption(f"{len(crudo_lote):,} pacientes · {reconocidas} de {len(VARIABLES)} variables del modelo reconocidas")
        if reconocidas == 0:
            st.error("Ninguna columna coincide con las variables del modelo. Revise los nombres con la plantilla.")
        if faltan_cols and reconocidas:
            st.info(f"Columnas que no vienen y quedan como Sin dato: {', '.join(faltan_cols)}")
        if desconocidos:
            st.warning("Valores no vistos en el entrenamiento (el modelo los trata como categorías poco frecuentes): "
                       + "; ".join(f"{etiqueta(c)}: {', '.join(map(str, v[:5]))}" for c, v in desconocidos.items()))
        if fuera:
            st.warning("Valores fuera del rango del entrenamiento: "
                       + "; ".join(f"{etiqueta(c)}: {n} pacientes" for c, n in fuera.items()))
        with st.expander("Vista previa del archivo"):
            st.dataframe(crudo_lote.head(20), hide_index=True)

        if st.button("Predecir lote", type="primary", disabled=reconocidas == 0):
            with st.spinner("Calculando el riesgo de cada paciente..."):
                st.session_state["lote"] = {"clave": clave_lote, "resultado": proc.predecir_lote(modelo, crudo_lote, meta)}

        estado_lote = st.session_state.get("lote")
        if estado_lote is not None and estado_lote["clave"] == clave_lote:
            resultado = estado_lote["resultado"]
            niveles = resultado["nivel"].value_counts()
            st.write("")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                tarjeta("Pacientes evaluados", f"{len(resultado):,}", f"Riesgo medio {resultado['riesgo_evento_12m'].mean():.1%}")
            with c2:
                tarjeta("Con alerta de alto riesgo", f"{resultado['alto_riesgo'].mean():.1%}", f"Umbral {meta['umbral']:.1%}", ROJO)
            with c3:
                tarjeta("Nivel alto", f"{niveles.get('Alto', 0):,}", f"Intermedio {niveles.get('Intermedio', 0):,}", OCRE)
            with c4:
                tarjeta("Nivel bajo", f"{niveles.get('Bajo', 0):,}", "Deciles 1 a 3 de df_test", VERDE)
            st.write("")
            fig = go.Figure(go.Histogram(x=resultado["riesgo_evento_12m"], xbins=dict(start=0, end=1, size=0.025), marker_color=AZUL))
            fig.add_vline(x=meta["umbral"], line_color=ROJO, annotation_text="Umbral")
            fig.update_layout(xaxis=dict(tickformat=".0%", title="Riesgo predicho", range=[0, 1]), yaxis_title="Pacientes", bargap=0.05)
            lienzo(fig, 260)
            st.dataframe(resultado, hide_index=True, height=360, column_config={
                "riesgo_evento_12m": st.column_config.ProgressColumn("riesgo_evento_12m", format="percent", min_value=0, max_value=1)})
            st.download_button("Descargar resultados (CSV)", a_csv(resultado), file_name="riesgo_post_trasplante.csv",
                               mime="text/csv", on_click="ignore", type="primary")
        else:
            st.caption("Pulse **Predecir lote** para calcular el riesgo de estos pacientes.")


# Pestaña: cómo funciona el modelo

with tabs["Cómo funciona el modelo"]:
    metricas = meta["metricas_prueba"]
    st.subheader("Desempeño en df_test: 4,285 pacientes que el modelo no vio al elegirse")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        tarjeta("AUC-ROC", f"{metricas['AUC-ROC']:.3f}", "Capacidad de ordenar pacientes por riesgo")
    with c2:
        tarjeta("Recall del evento", f"{metricas['recall del evento']:.1%}", "Eventos detectados con el umbral")
    with c3:
        tarjeta("Precision del evento", f"{metricas['precision del evento']:.1%}", "Alertas que resultan correctas")
    with c4:
        tarjeta("Brier score", f"{metricas['Brier score']:.3f}", "0.25 equivale a predecir siempre 50%")
    st.write("")
    izq, der = st.columns(2)
    with izq:
        st.markdown("**Riesgo predicho y tasa real por decil (df_test)**")
        lienzo(grafico_deciles(), 380)
    with der:
        st.markdown("**Las 15 variables que más pesan según SHAP (sección 12 del notebook)**")
        imp = pd.Series(meta["importancia_shap"]).sort_values(ascending=False).head(15).iloc[::-1]
        fig = go.Figure(go.Bar(x=imp.values, y=[etiqueta(c) for c in imp.index], orientation="h", marker_color=AZUL))
        fig.update_layout(xaxis_title="SHAP medio absoluto (log-odds)")
        lienzo(fig, 380)

    hp = meta["hiperparametros"]
    st.markdown(f"""
**Modelo desplegado.** {meta['modelo']} con `max_depth={hp['max_depth']}`, `learning_rate={hp['learning_rate']}`,
`min_child_weight={hp['min_child_weight']}` y `n_estimators={hp['n_estimators']}` (elegidos con GridSearchCV), dentro de un
Pipeline con el preprocesamiento. Umbral de alerta {meta['umbral']:.3f}, elegido con el índice de Youden en df_val. Para el
despliegue se reentrenó con los {meta['pacientes_despliegue']:,} pacientes de df_model; las métricas de arriba son las de df_test
antes de ese reentrenamiento. Average precision {metricas['average precision']:.3f}, F1 macro {metricas['F1 macro']:.3f} y
balanced accuracy {metricas['balanced accuracy']:.3f}.
""")
    with st.expander("Datos, método y límites"):
        st.markdown(f"""
**Datos.** Competencia CIBMTR, Equity in post-HCT Survival Predictions, de Kaggle ({meta['fuente_diccionario']}): 28,800
pacientes y 57 variables registradas antes del trasplante. El evento es `efs = 1` dentro de los primeros 12 meses; se excluyen
los 238 pacientes sin evento que salieron del seguimiento antes del año.

**Método.** Variables con más de 50% de nulos eliminadas ({', '.join(meta['columnas_eliminadas'])}), vacíos genéticos marcados
como No aplica y cinco variables nuevas; SimpleImputer, StandardScaler, OneHotEncoder y PCA sobre el bloque HLA dentro de un
Pipeline; seis modelos comparados con StratifiedKFold de 10 pliegues; GridSearchCV; umbral fijado en df_val; evaluación única en
df_test; explicabilidad con SHAP.

**Límites.** Los trasplantes son de 2008 a 2020. En niños y en injertos de médula ósea el umbral general detecta pocos eventos
porque su riesgo base es menor. Para usarlo en otro país o centro conviene recalibrarlo con datos locales.
""")
    with st.expander("Diccionario de variables del formulario"):
        diccionario = pd.DataFrame({
            "Variable": [etiqueta(c) for c in VARIABLES],
            "Nombre en el CSV (Kaggle)": [proc.NOMBRES_ORIGINALES.get(c, c) for c in VARIABLES],
            "Nombre en español": VARIABLES,
            "Descripción oficial": [meta["descripciones"].get(c, "") for c in VARIABLES],
            "Valores válidos": [f"{meta['rangos'][c][0]:g} a {meta['rangos'][c][2]:g}" if c in meta["rangos"]
                                else " | ".join(meta["opciones"][c]) for c in VARIABLES],
            "Si viene vacía": [texto_imputacion(c) for c in VARIABLES],
        })
        st.dataframe(diccionario, hide_index=True, height=420)
