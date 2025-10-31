#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dashboard principal para análisis bibliométrico.
Integra los requerimientos 1, 2 y 3 en una interfaz unificada.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import bibtexparser
from matplotlib import pyplot as plt
from matplotlib_venn import venn2
import io
import base64

# Importar módulos propios
from utils.extractor_bib import ExtratorBibTeX
from requerimiento2.similitud_textos import SimilitudTextos
from requerimiento3.frecuencia_palabras import AnalizadorFrecuencia
from requerimiento1.acm_descarga import ACMDescarga
from requerimiento1.scienciedirect import ScienceDirectDescarga
from requerimiento1.unir_bib_deduplicado import UnificadorBibTeX

def verificar_archivos():
    """Verifica la existencia de archivos y retorna estado."""
    archivos = {
        "ACM": Path("requerimiento1/descargas/acm/acmCompleto.bib"),
        "ScienceDirect": Path("requerimiento1/descargas/ScienceDirect/sciencedirectCompleto.bib"),
        "Unificado": Path("requerimiento1/descargas/resultado_unificado.bib")
    }
    return {nombre: ruta.exists() for nombre, ruta in archivos.items()}

# Configuración de la página
st.set_page_config(
    page_title="Análisis Bibliométrico UQ",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Funciones auxiliares
def cargar_bibliografia():
    """Carga el archivo BibTeX unificado."""
    ruta_bib = Path("requerimiento1/descargas/resultado_unificado.bib")
    if not ruta_bib.exists():
        return None
    
    with open(ruta_bib, 'r', encoding='utf-8') as archivo:
        parser = bibtexparser.bparser.BibTexParser()
        return bibtexparser.load(archivo, parser=parser)

def plot_to_base64(fig):
    """Convierte un plot de matplotlib a base64."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()

def explicar_algoritmo(nombre):
    """Retorna la explicación matemática y algorítmica de cada método."""
    explicaciones = {
        "Levenshtein": """
        ### Distancia de Levenshtein 📏
        
        La distancia de Levenshtein mide el número mínimo de operaciones necesarias para transformar una cadena en otra.
        
        #### Operaciones permitidas:
        1. Inserción (costo = 1)
        2. Eliminación (costo = 1)
        3. Sustitución (costo = 1 o 2)
        
        #### Fórmula matemática:
        
        Para dos cadenas a, b de longitud i, j:
        
        lev(i,j) = min(
            lev(i-1,j) + 1,              # eliminación
            lev(i,j-1) + 1,              # inserción
            lev(i-1,j-1) + costo_sust    # sustitución
        )
        
        donde costo_sust = 0 si a[i] = b[j], 2 en otro caso
        
        #### Complejidad:
        - Tiempo: O(mn)
        - Espacio: O(mn)
        donde m,n son las longitudes de las cadenas
        """,
        
        "Jaccard": """
        ### Índice de Jaccard 🔄
        
        Mide la similitud entre conjuntos comparando elementos comunes vs totales.
        
        #### Fórmula matemática:
        
        J(A,B) = |A ∩ B| / |A ∪ B|
        
        Donde:
        - A ∩ B: elementos comunes
        - A ∪ B: todos los elementos únicos
        
        #### Propiedades:
        - Rango: [0,1]
        - 0: conjuntos disjuntos
        - 1: conjuntos idénticos
        
        #### Complejidad:
        - Tiempo: O(n)
        - Espacio: O(n)
        donde n es el total de elementos únicos
        """,
        
        "TF-IDF Coseno": """
        ### Similitud Coseno con TF-IDF 📊
        
        Combina la ponderación TF-IDF con la similitud del coseno.
        
        #### Fórmulas:
        
        1. TF (Term Frequency):
           tf(t,d) = (frecuencia de t en d)
        
        2. IDF (Inverse Document Frequency):
           idf(t) = log(N/df_t)
           donde:
           - N: número total de documentos
           - df_t: documentos que contienen t
        
        3. Similitud Coseno:
           cos(A,B) = (A·B)/(||A||·||B||)
        
        #### Proceso:
        1. Calcular matriz TF-IDF
        2. Normalizar vectores
        3. Calcular coseno
        
        #### Complejidad:
        - Tiempo: O(nm)
        - Espacio: O(nm)
        donde n=términos, m=documentos
        """,
        
        "N-gramas": """
        ### Similitud por N-gramas 🔤
        
        Compara secuencias de n caracteres o palabras consecutivas.
        
        #### Proceso:
        1. Generar n-gramas:
           texto: "hello"
           3-gramas: ["hel", "ell", "llo"]
        
        2. Calcular similitud:
           sim = |común| / |total|
        
        #### Ventajas:
        - Resistente a errores menores
        - Captura patrones locales
        - Flexible en n
        
        #### Complejidad:
        - Tiempo: O(n-m+1)
        - Espacio: O(n-m+1)
        donde n=longitud texto, m=tamaño n-grama
        """,
        
        "Semantic Embedding": """
        ### Embedding Semántico 🧠
        
        Convierte texto en vectores que capturan significado semántico.
        
        #### Proceso:
        1. Tokenización
        2. Vectorización de palabras
        3. Agregación de vectores
        4. Similitud coseno
        
        #### Características:
        - Captura semántica
        - Independiente de orden exacto
        - Robusto a sinónimos
        
        #### Ventajas:
        - Entiende significado
        - Maneja variaciones
        - Escalable
        
        #### Complejidad:
        - Tiempo: O(n·d)
        - Espacio: O(d)
        donde n=palabras, d=dimensiones
        """,
        
        "Contextual Similarity": """
        ### Similitud Contextual 🌍
        
        Analiza similitud considerando contexto y relaciones.
        
        #### Componentes:
        1. Análisis de contexto local
        2. Ponderación por relevancia
        3. Agregación contextual
        
        #### Proceso:
        1. Identificar contextos
        2. Calcular similitudes locales
        3. Ponderar por importancia
        4. Agregar resultados
        
        #### Ventajas:
        - Sensible al contexto
        - Maneja ambigüedad
        - Resultados interpretables
        
        #### Complejidad:
        - Tiempo: O(n²)
        - Espacio: O(n)
        donde n=términos en contexto
        """
    }
    return explicaciones.get(nombre, "Explicación no disponible")

def mostrar_header():
    """Muestra el encabezado de la aplicación."""
    st.title("📚 Sistema de Análisis Bibliométrico UQ")
    st.markdown("""
    Sistema integrado para análisis bibliométrico que incluye:
    - Gestión de referencias bibliográficas
    - Análisis de similitud textual
    - Análisis de frecuencia de palabras
    """)

def mostrar_estado_actual():
    """Muestra el estado actual de los archivos bibliográficos."""
    st.subheader("📁 Estado Actual")
    
    # Verificar archivos
    archivos = {
        "ACM": Path("requerimiento1/descargas/acm/acmCompleto.bib"),
        "ScienceDirect": Path("requerimiento1/descargas/ScienceDirect/sciencedirectCompleto.bib"),
        "Unificado": Path("requerimiento1/descargas/resultado_unificado.bib")
    }
    
    for nombre, ruta in archivos.items():
        if ruta.exists():
            with open(ruta, 'r', encoding='utf-8') as f:
                db = bibtexparser.load(f)
                st.success(f"✅ {nombre}: {len(db.entries)} referencias")
        else:
            st.warning(f"⚠️ {nombre}: Archivo no encontrado")

@st.cache_data
def analizar_similitud_articulos(texto1, texto2, metodo):
    """Analiza la similitud entre dos textos."""
    similitud = SimilitudTextos()
    metodos = {
        "Levenshtein": similitud.levenshtein,
        "Jaccard": similitud.jaccard,
        "TF-IDF Coseno": similitud.tfidf_coseno,
        "N-gramas": similitud.ngramas,
        "Semantic Embedding": similitud.semantic_embedding,
        "Contextual Similarity": similitud.contextual_similarity
    }
    return metodos[metodo](texto1, texto2)

def visualizar_similitud(score, metodo):
    """Visualiza el score de similitud con un gauge."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        title={'text': f"Similitud {metodo}"},
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
    return fig

def ejecutar_scraping(fuente):
    """Ejecuta el scraping para la fuente especificada."""
    if fuente == "ACM":
        with st.spinner("Extrayendo referencias de ACM..."):
            extractor = ACMDescarga()
            try:
                extractor.abrir_base_datos()
                return True
            except Exception as e:
                st.error(f"Error: {str(e)}")
                return False
            finally:
                extractor.cerrar()
    elif fuente == "ScienceDirect":
        with st.spinner("Extrayendo referencias de ScienceDirect..."):
            extractor = ScienceDirectDescarga()
            try:
                extractor.abrir_base_datos()
                return True
            except Exception as e:
                st.error(f"Error: {str(e)}")
                return False
            finally:
                extractor.cerrar()
    return False

def mostrar_seccion_scraping():
    """Muestra la sección de scraping con botones condicionales."""
    st.header("🌐 Web Scraping de Referencias")
    
    # Verificar estado de archivos
    estados = verificar_archivos()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ACM Digital Library")
        boton_acm = st.button(
            "Extraer de ACM",
            disabled=estados["ACM"],
            help="Extraer referencias de ACM Digital Library"
        )
        if estados["ACM"]:
            st.success("✅ Referencias de ACM ya extraídas")
        if boton_acm:
            if ejecutar_scraping("ACM"):
                st.success("✅ Referencias extraídas exitosamente")
                estados["ACM"] = True
            else:
                st.error("❌ Error en la extracción")
    
    with col2:
        st.subheader("ScienceDirect")
        boton_sd = st.button(
            "Extraer de ScienceDirect",
            disabled=estados["ScienceDirect"],
            help="Extraer referencias de ScienceDirect"
        )
        if estados["ScienceDirect"]:
            st.success("✅ Referencias de ScienceDirect ya extraídas")
        if boton_sd:
            if ejecutar_scraping("ScienceDirect"):
                st.success("✅ Referencias extraídas exitosamente")
                estados["ScienceDirect"] = True
            else:
                st.error("❌ Error en la extracción")
    
    # Unificación
    st.subheader("Unificación de Referencias")
    boton_unificar = st.button(
        "Unificar Referencias",
        disabled=estados["Unificado"] or not (estados["ACM"] and estados["ScienceDirect"]),
        help="Unificar y deduplicar referencias de ambas fuentes"
    )
    
    if estados["Unificado"]:
        st.success("✅ Referencias ya unificadas")
    elif boton_unificar:
        with st.spinner("Unificando referencias..."):
            unificador = UnificadorBibTeX()
            try:
                if unificador.unificar():
                    st.success("✅ Referencias unificadas exitosamente")
                    estados["Unificado"] = True
                else:
                    st.error("❌ Error en la unificación")
            except Exception as e:
                st.error(f"❌ Error en la unificación: {str(e)}")

def main():
    """Función principal."""
    mostrar_header()
    
    # Sistema de pestañas principales
    tab1, tab2, tab3 = st.tabs([
        "🌐 Web Scraping",
        "🔄 Análisis de Similitud",
        "📊 Análisis de Frecuencia"
    ])
    
    # Pestaña 1: Web Scraping
    with tab1:
        mostrar_seccion_scraping()
    
    # Pestaña 2: Análisis de Similitud
    with tab2:
        st.header("🔄 Análisis de Similitud")
        
        # Cargar bibliografía
        bibliografia = cargar_bibliografia()
        if bibliografia is None:
            st.error("❌ No se encontró el archivo de referencias unificado")
            return
        
        # Selector de artículos
        articulos = [(entry.get('ID', ''), entry.get('title', '')) 
                    for entry in bibliografia.entries 
                    if 'abstract' in entry]
        
        col1, col2 = st.columns(2)
        
        with col1:
            art1 = st.selectbox(
                "Seleccionar primer artículo",
                options=articulos,
                format_func=lambda x: x[1]
            )
        
        with col2:
            art2 = st.selectbox(
                "Seleccionar segundo artículo",
                options=articulos,
                format_func=lambda x: x[1]
            )
        
        if art1 and art2:
            # Obtener abstracts
            abstract1 = next(e['abstract'] for e in bibliografia.entries if e['ID'] == art1[0])
            abstract2 = next(e['abstract'] for e in bibliografia.entries if e['ID'] == art2[0])
            
            # Mostrar abstracts
            with st.expander("Ver Abstracts"):
                st.markdown("**Abstract 1:**")
                st.write(abstract1)
                st.markdown("**Abstract 2:**")
                st.write(abstract2)
            
            # Analizar similitud con todos los métodos
            metodos = [
                "Levenshtein",
                "Jaccard",
                "TF-IDF Coseno",
                "N-gramas",
                "Semantic Embedding",
                "Contextual Similarity"
            ]
            
            # Crear tabs para cada método
            method_tabs = st.tabs(metodos)
            
            resultados = {}
            for metodo, tab in zip(metodos, method_tabs):
                with tab:
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown(explicar_algoritmo(metodo))
                    
                    with col2:
                        with st.spinner(f"Calculando similitud con {metodo}..."):
                            score = analizar_similitud_articulos(abstract1, abstract2, metodo)
                            resultados[metodo] = score
                            st.plotly_chart(
                                visualizar_similitud(score, metodo),
                                use_container_width=True
                            )
            
            # Mostrar comparativa
            st.header("📊 Comparativa de Métodos")
            
            # Gráfico de radar
            categorias = list(resultados.keys())
            valores = list(resultados.values())
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=valores,
                theta=categorias,
                fill='toself',
                name='Similitud'
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Pestaña 3: Análisis de Frecuencia
    with tab3:
        st.header("📊 Análisis de Frecuencia")
        
        # Verificar archivo unificado
        ruta_bib = Path("requerimiento1/descargas/resultado_unificado.bib")
        if not ruta_bib.exists():
            st.error("❌ No se encontró el archivo de referencias unificado. Por favor, complete el proceso de scraping primero.")
            return
            
        try:
            analizador = AnalizadorFrecuencia(str(ruta_bib))
            
            # Configuración
            col1, col2 = st.columns(2)
            with col1:
                min_df = st.slider(
                    "Frecuencia mínima de documento",
                    min_value=0.01,
                    max_value=0.5,
                    value=0.05,
                    step=0.01,
                    help="Proporción mínima de documentos en los que debe aparecer una palabra"
                )
            
            with col2:
                incluir_bigramas = st.checkbox(
                    "Incluir bigramas",
                    value=True,
                    help="Analizar pares de palabras además de palabras individuales"
                )
            
            if st.button("Realizar Análisis", key="analisis_frecuencia"):
                with st.spinner("Analizando textos..."):
                    # Frecuencias predefinidas
                    df_freq = analizador.contar_palabras_predefinidas()
                    
                    # TF-IDF
                    df_tfidf = analizador.extraer_palabras_tfidf(
                        min_df=min_df,
                        incluir_bigramas=incluir_bigramas
                    )
                    
                    # Visualizaciones
                    st.subheader("📈 Análisis de Palabras Predefinidas")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_freq = px.bar(
                            df_freq.head(10),
                            x='palabra',
                            y='frecuencia_total',
                            title='Top 10 Palabras más Frecuentes',
                            labels={
                                'palabra': 'Palabra',
                                'frecuencia_total': 'Frecuencia Total'
                            }
                        )
                        fig_freq.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig_freq, use_container_width=True)
                    
                    with col2:
                        fig_docs = px.bar(
                            df_freq.head(10),
                            x='palabra',
                            y='porcentaje_docs',
                            title='Cobertura en Documentos',
                            labels={
                                'palabra': 'Palabra',
                                'porcentaje_docs': 'Porcentaje de Documentos (%)'
                            }
                        )
                        fig_docs.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig_docs, use_container_width=True)
                    
                    st.markdown("##### Detalles de Frecuencias")
                    st.dataframe(
                        df_freq.style.format({
                            'porcentaje_docs': '{:.2f}%'
                        })
                    )
                    
                    st.markdown("---")
                    
                    st.subheader("🔍 Análisis TF-IDF")
                    col1, col2 = st.columns(2)
                    
                    with col1:
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
                        fig_rel = px.pie(
                            df_tfidf.head(10),
                            values='freq_relativa',
                            names='palabra',
                            title='Distribución de Frecuencia Relativa'
                        )
                        st.plotly_chart(fig_rel, use_container_width=True)
                    
                    st.markdown("##### Métricas Detalladas")
                    st.dataframe(
                        df_tfidf.style.format({
                            'score_tfidf': '{:.4f}',
                            'freq_relativa': '{:.2%}',
                            'score_combinado': '{:.4f}'
                        })
                    )
        
        except Exception as e:
            st.error(f"Error al realizar el análisis: {str(e)}")
            st.error("Por favor, asegúrese de que el archivo de referencias está correctamente formateado y contiene abstracts.")

if __name__ == "__main__":
    main()
