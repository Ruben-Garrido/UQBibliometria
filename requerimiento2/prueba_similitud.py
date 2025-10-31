#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script de prueba para los algoritmos de similitud."""

from similitud_textos import SimilitudTextos

def probar_similitud():
    """Prueba todos los algoritmos de similitud con textos de ejemplo."""
    
    # Textos de ejemplo con más contexto
    textos = [
        """Los algoritmos de machine learning son fundamentales en el análisis de datos.
        Las técnicas modernas permiten procesar grandes volúmenes de información y extraer patrones.""",
        
        """El aprendizaje automático es esencial para analizar grandes cantidades de datos.
        Los modelos aprenden patrones y pueden hacer predicciones basadas en información histórica.""",
        
        """La física cuántica estudia el comportamiento de partículas subatómicas.
        Esta rama de la física describe fenómenos a escala microscópica."""  # Texto diferente
    ]
    
    # Crear instancia del comparador
    similitud = SimilitudTextos()
    
    # Probar cada algoritmo
    for nombre in similitud.ALGORITMOS:
        print(f"\nAlgoritmo: {nombre}")
        print("-" * 50)
        
        # Comparar cada par de textos
        for i in range(len(textos)):
            for j in range(i + 1, len(textos)):
                try:
                    valor = similitud.calcular_similitud(textos[i], textos[j], nombre)
                    print(f"Texto {i+1} vs Texto {j+1}: {valor:.4f}")
                except Exception as e:
                    print(f"Error al comparar Texto {i+1} vs Texto {j+1}: {e}")

if __name__ == "__main__":
    probar_similitud()