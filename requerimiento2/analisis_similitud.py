#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para análisis de similitud entre múltiples abstracts.
Implementa comparación de textos y generación de matriz de similitud.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import json

from similitud_textos import SimilitudTextos
from utils.extractor_bib import ExtratorBibTeX
from utils.exportador import Exportador

class AnalizadorSimilitud:
    """
    Clase para analizar similitud entre múltiples textos y generar matriz de similitud.
    """
    
    def __init__(self, ruta_bib: Optional[str] = None):
        """
        Inicializa el analizador de similitud.
        
        Args:
            ruta_bib: Ruta al archivo BibTeX (opcional)
        """
        self.ruta_bib = Path(ruta_bib) if ruta_bib else None
        self.similitud = SimilitudTextos()
        self._abstracts: List[Dict] = []
        
        if self.ruta_bib:
            self._cargar_abstracts()
    
    def _cargar_abstracts(self) -> None:
        """Carga los abstracts desde el archivo BibTeX."""
        if not self.ruta_bib or not self.ruta_bib.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {self.ruta_bib}")
            
        extractor = ExtratorBibTeX(self.ruta_bib)
        self._abstracts = extractor.get_abstracts()
        
    def comparar_textos(self, 
                       textos: List[str], 
                       algoritmo: str = 'tfidf_coseno') -> pd.DataFrame:
        """
        Compara una lista de textos entre sí.
        
        Args:
            textos: Lista de textos a comparar
            algoritmo: Nombre del algoritmo a usar
            
        Returns:
            DataFrame con matriz de similitud
        """
        n = len(textos)
        matriz = np.zeros((n, n))
        
        # Calcular similitudes
        for i in range(n):
            for j in range(i, n):
                sim = self.similitud.calcular_similitud(textos[i], textos[j], algoritmo)
                matriz[i,j] = sim
                if i != j:
                    matriz[j,i] = sim
                    
        return pd.DataFrame(
            matriz,
            index=[f"Texto {i+1}" for i in range(n)],
            columns=[f"Texto {i+1}" for i in range(n)]
        )
    
    def analizar_abstracts(self, 
                          algoritmo: str = 'tfidf_coseno',
                          limite: Optional[int] = None) -> pd.DataFrame:
        """
        Analiza similitud entre abstracts del archivo BibTeX.
        
        Args:
            algoritmo: Nombre del algoritmo a usar
            limite: Número máximo de abstracts a comparar
            
        Returns:
            DataFrame con matriz de similitud
        """
        if not self._abstracts:
            raise ValueError("No hay abstracts cargados")
            
        # Limitar cantidad de abstracts si se especifica
        abstracts = self._abstracts[:limite] if limite else self._abstracts
        
        # Extraer textos y IDs
        textos = [a["abstract"] for a in abstracts]
        ids = [a["id"] for a in abstracts]
        
        # Calcular matriz de similitud
        matriz = self.comparar_textos(textos, algoritmo)
        matriz.index = ids
        matriz.columns = ids
        
        return matriz
    
    def exportar_resultados(self,
                           matriz: pd.DataFrame,
                           directorio: str,
                           prefijo: str = "similitud") -> Dict[str, Path]:
        """
        Exporta resultados en diferentes formatos.
        
        Args:
            matriz: DataFrame con matriz de similitud
            directorio: Directorio donde guardar archivos
            prefijo: Prefijo para nombres de archivo
            
        Returns:
            Dict con rutas a los archivos generados
        """
        dir_salida = Path(directorio)
        dir_salida.mkdir(parents=True, exist_ok=True)
        
        archivos = {}
        
        # Exportar CSV
        ruta_csv = dir_salida / f"{prefijo}_matriz.csv"
        archivos['csv'] = Exportador.a_csv(matriz, ruta_csv)
        
        # Exportar JSON
        datos_json = {
            'matriz': matriz.to_dict(),
            'metadata': {
                'dimensiones': matriz.shape,
                'ids': list(matriz.index)
            }
        }
        ruta_json = dir_salida / f"{prefijo}_datos.json"
        archivos['json'] = Exportador.a_json(datos_json, ruta_json)
        
        return archivos

def main():
    """Función principal para pruebas."""
    # Ruta al archivo BibTeX unificado
    ruta_bib = Path(__file__).parents[1] / "requerimiento1" / "descargas" / "resultado_unificado.bib"
    
    try:
        # Crear analizador
        analizador = AnalizadorSimilitud(ruta_bib)
        
        # Analizar con diferentes algoritmos
        for algoritmo in SimilitudTextos.ALGORITMOS:
            print(f"\nAnalizando con {algoritmo}...")
            matriz = analizador.analizar_abstracts(algoritmo=algoritmo, limite=5)
            print("\nMatriz de similitud:")
            print(matriz)
            
            # Exportar resultados
            dir_salida = Path(__file__).parent / "resultados" / algoritmo
            archivos = analizador.exportar_resultados(matriz, dir_salida, algoritmo)
            print("\nArchivos generados:")
            for fmt, ruta in archivos.items():
                print(f"- {fmt}: {ruta}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()