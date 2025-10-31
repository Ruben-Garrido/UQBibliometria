#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para análisis de frecuencia de palabras en abstracts.
"""

from collections import Counter
from typing import List, Dict, Set, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re

from utils.extractor_bib import ExtratorBibTeX
from utils.preprocesador_texto import PreprocesadorTexto
from utils.exportador import Exportador

class AnalizadorFrecuencia:
    """Clase para análisis de frecuencia de palabras en textos académicos."""
    
    # Lista predefinida de palabras de interés
    PALABRAS_INTERES = {
        'algorithm', 'optimization', 'complexity', 'performance', 'efficiency',
        'analysis', 'computation', 'solution', 'problem', 'method',
        'implementation', 'system', 'time', 'cost', 'approach'
    }
    
    def __init__(self, ruta_bib: Optional[str] = None):
        """
        Inicializa el analizador de frecuencia.
        
        Args:
            ruta_bib: Ruta al archivo BibTeX (opcional)
        """
        self.ruta_bib = ruta_bib
        self.preprocesador = PreprocesadorTexto()
        self._abstracts = []
        
        if ruta_bib:
            self._cargar_abstracts()
    
    def _cargar_abstracts(self) -> None:
        """Carga los abstracts desde el archivo BibTeX."""
        extractor = ExtratorBibTeX(self.ruta_bib)
        self._abstracts = extractor.get_abstracts()
    
    def contar_palabras_predefinidas(self) -> pd.DataFrame:
        """
        Cuenta la frecuencia de las palabras predefinidas en los abstracts.
        
        Returns:
            DataFrame con frecuencias de palabras predefinidas
        """
        if not self._abstracts:
            raise ValueError("No hay abstracts cargados")
            
        # Procesar todos los abstracts
        frecuencias = Counter()
        total_docs = len(self._abstracts)
        docs_por_palabra = Counter()
        
        for abstract in self._abstracts:
            texto = abstract.get("abstract", "").lower()
            tokens = set(self.preprocesador.procesar_texto(texto))
            
            # Contar apariciones por documento
            for palabra in self.PALABRAS_INTERES:
                if palabra in tokens:
                    docs_por_palabra[palabra] += 1
                    frecuencias[palabra] += texto.count(palabra)
        
        # Crear DataFrame con métricas
        resultados = []
        for palabra in self.PALABRAS_INTERES:
            frec = frecuencias[palabra]
            docs = docs_por_palabra[palabra]
            resultados.append({
                'palabra': palabra,
                'frecuencia_total': frec,
                'documentos': docs,
                'porcentaje_docs': (docs / total_docs) * 100 if total_docs > 0 else 0
            })
        
        df = pd.DataFrame(resultados)
        return df.sort_values('frecuencia_total', ascending=False)
    
    def extraer_palabras_tfidf(self, 
                                  n_palabras: int = 15, 
                                  min_df: float = 0.05,
                                  incluir_bigramas: bool = True) -> pd.DataFrame:
        """
        Extrae palabras clave usando TF-IDF con mejoras.
        
        Args:
            n_palabras: Número de palabras a extraer
            min_df: Frecuencia mínima de documento (entre 0 y 1)
            incluir_bigramas: Si True, incluye bigramas en el análisis
            
        Returns:
            DataFrame con palabras clave ordenadas por importancia TF-IDF
        """
        if not self._abstracts:
            raise ValueError("No hay abstracts cargados")
            
        # Preparar textos
        textos = [abstract.get("abstract", "") for abstract in self._abstracts]
        
        def tokenizer(texto):
            return self.preprocesador.procesar_texto(
                texto, 
                incluir_bigramas=incluir_bigramas
            )
        
        # Configurar vectorizador TF-IDF con parámetros optimizados
        vectorizer = TfidfVectorizer(
            tokenizer=tokenizer,
            min_df=min_df,  # Reducido para capturar términos menos frecuentes
            max_df=0.95,    # Eliminar términos que aparecen en casi todos los docs
            max_features=1000,  # Aumentado para considerar más términos
            ngram_range=(1, 1) if not incluir_bigramas else (1, 2),
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True  # Aplicar escala sublinear al TF
        )
        
        # Calcular TF-IDF
        tfidf_matrix = vectorizer.fit_transform(textos)
        
        # Calcular scores promedio ponderados por documento
        doc_lengths = tfidf_matrix.sum(axis=1).A1
        weights = doc_lengths / doc_lengths.sum()
        scores_promedio = np.array(
            np.average(tfidf_matrix.toarray(), axis=0, weights=weights)
        )
        
        palabras = vectorizer.get_feature_names_out()
        
        # Crear DataFrame con métricas adicionales
        resultados = pd.DataFrame({
            'palabra': palabras,
            'score_tfidf': scores_promedio,
            'docs_aparicion': (tfidf_matrix > 0).sum(axis=0).A1,
            'freq_relativa': (tfidf_matrix > 0).sum(axis=0).A1 / len(textos)
        })
        
        # Ordenar por score TF-IDF y frecuencia relativa
        resultados['score_combinado'] = (
            0.7 * resultados['score_tfidf'] + 
            0.3 * resultados['freq_relativa']
        )
        resultados = resultados.sort_values('score_combinado', ascending=False)
        
        return resultados.head(n_palabras)
    
    def calcular_metricas_precision(self,
                                  palabras_extraidas: Set[str],
                                  tolerancia: float = 0.8) -> Dict[str, float]:
        """
        Calcula métricas de precisión para palabras extraídas.
        
        Args:
            palabras_extraidas: Conjunto de palabras extraídas
            tolerancia: Umbral de similitud para considerar match
            
        Returns:
            Diccionario con métricas de precisión
        """
        if not palabras_extraidas:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
            
        # Usar similitud para encontrar matches
        true_positives = 0
        for palabra_extraida in palabras_extraidas:
            for palabra_referencia in self.PALABRAS_INTERES:
                # Similitud simple por ahora
                if palabra_extraida in palabra_referencia or palabra_referencia in palabra_extraida:
                    true_positives += 1
                    break
                    
        precision = true_positives / len(palabras_extraidas)
        recall = true_positives / len(self.PALABRAS_INTERES)
        
        f1_score = 0.0
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
            
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    
    def exportar_resultados(self,
                          df_frecuencias: pd.DataFrame,
                          df_tfidf: pd.DataFrame,
                          metricas: Dict[str, float],
                          directorio: str) -> Dict[str, str]:
        """
        Exporta resultados del análisis en diferentes formatos.
        
        Args:
            df_frecuencias: DataFrame con frecuencias predefinidas
            df_tfidf: DataFrame con palabras extraídas
            metricas: Diccionario con métricas de precisión
            directorio: Directorio donde guardar resultados
            
        Returns:
            Diccionario con rutas a los archivos generados
        """
        dir_salida = Path(directorio)
        dir_salida.mkdir(parents=True, exist_ok=True)
        
        archivos = {}
        
        # Exportar CSV de frecuencias
        ruta_csv = dir_salida / "frecuencias_palabras.csv"
        archivos['frecuencias_csv'] = str(
            Exportador.a_csv(df_frecuencias, ruta_csv)
        )
        
        # Exportar JSON con palabras TF-IDF y métricas
        datos_json = {
            'palabras_tfidf': df_tfidf.to_dict(orient='records'),
            'metricas_precision': metricas
        }
        ruta_json = dir_salida / "analisis_palabras.json"
        archivos['analisis_json'] = str(
            Exportador.a_json(datos_json, ruta_json)
        )
        
        return archivos

def main():
    """Función principal para pruebas."""
    # Ruta al archivo BibTeX unificado
    ruta_bib = Path(__file__).parents[1] / "requerimiento1" / "descargas" / "resultado_unificado.bib"
    
    try:
        # Crear analizador
        analizador = AnalizadorFrecuencia(ruta_bib)
        
        # 1. Análisis de palabras predefinidas
        print("\nAnálisis de palabras predefinidas:")
        df_frecuencias = analizador.contar_palabras_predefinidas()
        print(df_frecuencias)
        
        # 2. Extracción de palabras con TF-IDF
        print("\nPalabras extraídas con TF-IDF:")
        df_tfidf = analizador.extraer_palabras_tfidf()
        print(df_tfidf)
        
        # 3. Calcular métricas de precisión
        palabras_extraidas = set(df_tfidf['palabra'].values)
        metricas = analizador.calcular_metricas_precision(palabras_extraidas)
        print("\nMétricas de precisión:")
        for metrica, valor in metricas.items():
            print(f"{metrica}: {valor:.4f}")
            
        # 4. Exportar resultados
        dir_salida = Path(__file__).parent / "resultados"
        archivos = analizador.exportar_resultados(
            df_frecuencias, df_tfidf, metricas, dir_salida
        )
        print("\nArchivos generados:")
        for tipo, ruta in archivos.items():
            print(f"- {tipo}: {ruta}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()