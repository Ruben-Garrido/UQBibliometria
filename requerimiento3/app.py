#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interfaz web para el análisis de frecuencia de palabras.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from requerimiento3.frecuencia_palabras import AnalizadorFrecuencia

def configurar_pagina():
    """Configura la página de Streamlit."""
    st.set_page_config(
        page_title="Análisis de Frecuencia de Palabras",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Análisis de Frecuencia de Palabras")
    st.write("""
    Esta herramienta analiza la frecuencia de palabras en abstracts de artículos académicos.
    Utiliza tanto un conjunto predefinido de palabras clave como técnicas de TF-IDF para extraer
    términos relevantes.
    """)

def cargar_datos():
    """Carga y prepara los datos para el análisis."""
    # Ruta al archivo BibTeX unificado
    ruta_bib = Path(__file__).parents[1] / "requerimiento1" / "descargas" / "resultado_unificado.bib"
    
    if not ruta_bib.exists():
        st.error(f"No se encontró el archivo BibTeX en {ruta_bib}")
        st.stop()
    
    return AnalizadorFrecuencia(ruta_bib)

def mostrar_metricas_generales(analizador):
    """Muestra métricas generales del análisis."""
    st.header("📈 Métricas Generales")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Total de Documentos Analizados",
            value=len(analizador._abstracts)
        )
    
    with col2:
        palabras_interes = len(analizador.PALABRAS_INTERES)
        st.metric(
            label="Palabras Clave Monitoreadas",
            value=palabras_interes
        )

def visualizar_frecuencias_predefinidas(df_frecuencias):
    """Visualiza las frecuencias de palabras predefinidas."""
    st.header("🎯 Análisis de Palabras Predefinidas")
    
    # Gráfico de barras para frecuencias
    fig_freq = px.bar(
        df_frecuencias,
        x='palabra',
        y='frecuencia_total',
        title='Frecuencia Total por Palabra',
        labels={
            'palabra': 'Palabra',
            'frecuencia_total': 'Frecuencia Total'
        }
    )
    fig_freq.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_freq, use_container_width=True)
    
    # Gráfico de porcentaje de documentos
    fig_docs = px.bar(
        df_frecuencias,
        x='palabra',
        y='porcentaje_docs',
        title='Porcentaje de Documentos por Palabra',
        labels={
            'palabra': 'Palabra',
            'porcentaje_docs': 'Porcentaje de Documentos (%)'
        }
    )
    fig_docs.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_docs, use_container_width=True)
    
    # Tabla detallada
    st.subheader("📋 Detalles por Palabra")
    st.dataframe(
        df_frecuencias.style.format({
            'porcentaje_docs': '{:.2f}%'
        })
    )

def visualizar_analisis_tfidf(df_tfidf):
    """Visualiza los resultados del análisis TF-IDF."""
    st.header("🔍 Análisis TF-IDF")
    
    # Configuración de columnas
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Gráfico de scores TF-IDF
        fig_tfidf = px.bar(
            df_tfidf.head(10),
            x='palabra',
            y='score_tfidf',
            title='Top 10 Palabras por Score TF-IDF',
            labels={
                'palabra': 'Palabra',
                'score_tfidf': 'Score TF-IDF'
            }
        )
        fig_tfidf.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_tfidf, use_container_width=True)
    
    with col2:
        # Gráfico de pastel para frecuencia relativa
        fig_pie = px.pie(
            df_tfidf.head(10),
            values='freq_relativa',
            names='palabra',
            title='Distribución de Frecuencia Relativa (Top 10)'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Tabla de resultados completa
    st.subheader("📊 Resultados Detallados")
    st.dataframe(
        df_tfidf.style.format({
            'score_tfidf': '{:.4f}',
            'freq_relativa': '{:.2%}',
            'score_combinado': '{:.4f}'
        })
    )

def visualizar_metricas_precision(metricas):
    """Visualiza las métricas de precisión."""
    st.header("📌 Métricas de Precisión")
    
    # Crear gráfico de indicadores
    fig = go.Figure()
    
    metricas_info = [
        ("Precisión", metricas['precision'], "Proporción de palabras correctamente identificadas"),
        ("Recall", metricas['recall'], "Proporción de palabras relevantes identificadas"),
        ("F1-Score", metricas['f1_score'], "Media armónica de precisión y recall")
    ]
    
    for i, (nombre, valor, desc) in enumerate(metricas_info):
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=valor * 100,
            domain={'row': 0, 'column': i},
            title={'text': nombre, 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 33], 'color': "lightgray"},
                    {'range': [33, 66], 'color': "gray"},
                    {'range': [66, 100], 'color': "darkgray"}
                ],
            }
        ))
    
    fig.update_layout(
        grid={'rows': 1, 'columns': 3, 'pattern': "independent"},
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Explicación de métricas
    with st.expander("ℹ️ Explicación de Métricas"):
        st.write("""
        - **Precisión**: Mide qué proporción de las palabras identificadas son realmente relevantes.
        - **Recall**: Mide qué proporción de todas las palabras relevantes fueron identificadas.
        - **F1-Score**: Combina precisión y recall en una sola métrica, útil cuando se busca un balance entre ambas.
        """)

def main():
    """Función principal de la aplicación."""
    configurar_pagina()
    
    # Cargar datos
    with st.spinner("Cargando datos..."):
        analizador = cargar_datos()
    
    # Mostrar métricas generales
    mostrar_metricas_generales(analizador)
    
    # Separador
    st.markdown("---")
    
    # Configuración del análisis
    with st.expander("⚙️ Configuración del Análisis", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            min_df = st.slider(
                "Frecuencia mínima de documento",
                min_value=0.01,
                max_value=0.5,
                value=0.05,
                step=0.01,
                help="Mínima proporción de documentos en los que debe aparecer una palabra"
            )
            
        with col2:
            incluir_bigramas = st.checkbox(
                "Incluir bigramas",
                value=True,
                help="Analizar pares de palabras además de palabras individuales"
            )
    
    # Realizar análisis
    with st.spinner("Analizando documentos..."):
        # Análisis de palabras predefinidas
        df_frecuencias = analizador.contar_palabras_predefinidas()
        
        # Análisis TF-IDF
        df_tfidf = analizador.extraer_palabras_tfidf(
            min_df=min_df,
            incluir_bigramas=incluir_bigramas
        )
        
        # Calcular métricas
        palabras_extraidas = set(df_tfidf['palabra'].values)
        metricas = analizador.calcular_metricas_precision(palabras_extraidas)
    
    # Visualizar resultados
    tab1, tab2, tab3 = st.tabs([
        "📊 Frecuencias Predefinidas",
        "🔍 Análisis TF-IDF",
        "📈 Métricas de Precisión"
    ])
    
    with tab1:
        visualizar_frecuencias_predefinidas(df_frecuencias)
    
    with tab2:
        visualizar_analisis_tfidf(df_tfidf)
    
    with tab3:
        visualizar_metricas_precision(metricas)

if __name__ == "__main__":
    main()