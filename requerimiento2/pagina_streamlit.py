import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from requerimiento2.similitud_textos import SimilitudTextos
from requerimiento2.analisis_similitud import AnalizadorSimilitud

def mostrar_pagina_similitud():
    """Página de Streamlit para análisis de similitud de textos."""
    
    st.title("Análisis de Similitud de Textos")
    
    # Sidebar para opciones
    st.sidebar.header("Configuración")
    
    modo = st.sidebar.selectbox(
        "Modo de análisis",
        ["Comparar Textos", "Analizar Abstracts"],
        help="Seleccione si quiere comparar textos personalizados o analizar abstracts del BibTeX"
    )
    
    algoritmo = st.sidebar.selectbox(
        "Algoritmo de similitud",
        list(SimilitudTextos.ALGORITMOS.keys()),
        help="Seleccione el algoritmo para calcular similitud"
    )
    
    if modo == "Comparar Textos":
        st.header("Comparación de Textos")
        
        # Área para ingresar textos
        col1, col2 = st.columns(2)
        
        with col1:
            texto1 = st.text_area("Texto 1", height=200)
            
        with col2:
            texto2 = st.text_area("Texto 2", height=200)
            
        if texto1 and texto2:
            similitud = SimilitudTextos()
            valor = similitud.calcular_similitud(texto1, texto2, algoritmo)
            
            st.metric(
                label=f"Similitud ({algoritmo})",
                value=f"{valor:.4f}"
            )
            
            # Visualización como barra de progreso
            st.progress(valor)
            
    else:  # Analizar Abstracts
        st.header("Análisis de Abstracts")
        
        # Ruta al archivo BibTeX
        ruta_bib = Path(__file__).parents[1] / "requerimiento1" / "descargas" / "resultado_unificado.bib"
        
        if not ruta_bib.exists():
            st.error(f"No se encontró el archivo BibTeX en: {ruta_bib}")
            return
            
        try:
            # Configurar límite de abstracts
            limite = st.slider(
                "Número de abstracts a analizar",
                min_value=2,
                max_value=20,
                value=5,
                help="Seleccione cuántos abstracts comparar (más abstracts = más tiempo de procesamiento)"
            )
            
            # Analizar similitud
            analizador = AnalizadorSimilitud(ruta_bib)
            matriz = analizador.analizar_abstracts(algoritmo=algoritmo, limite=limite)
            
            # Mostrar matriz
            st.subheader("Matriz de Similitud")
            st.dataframe(matriz)
            
            # Visualización con heatmap
            st.subheader("Mapa de Calor")
            fig = px.imshow(
                matriz,
                labels=dict(x="Documento", y="Documento", color="Similitud"),
                color_continuous_scale="RdYlBu",
                aspect="auto"
            )
            st.plotly_chart(fig)
            
            # Exportar resultados
            if st.button("Exportar Resultados"):
                dir_salida = Path(__file__).parent / "resultados" / algoritmo
                archivos = analizador.exportar_resultados(matriz, dir_salida, algoritmo)
                
                st.success("Resultados exportados exitosamente:")
                for fmt, ruta in archivos.items():
                    st.write(f"- {fmt}: {ruta}")
                    
        except Exception as e:
            st.error(f"Error al analizar abstracts: {e}")

if __name__ == "__main__":
    mostrar_pagina_similitud()