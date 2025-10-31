#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementación de algoritmos de similitud de textos.
"""

from typing import Dict, List, Tuple, Callable, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import Levenshtein
from collections import defaultdict
import re

class SimilitudTextos:
    """Clase que implementa diferentes algoritmos de similitud de textos."""

    def __init__(self):
        """Inicializa recursos necesarios para los algoritmos."""
        # Stop words en español e inglés más comunes
        self.stop_words = {
            'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las',
            'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al', 'lo', 'como',
            'más', 'pero', 'sus', 'le', 'ya', 'o', 'este', 'sí', 'porque',
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her'
        }
            
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocesa el texto para análisis semántico.
        
        Args:
            text: Texto a preprocesar
            
        Returns:
            Lista de tokens preprocesados
        """
        # Convertir a minúsculas
        text = text.lower()
        
        # Remover caracteres especiales y números
        text = re.sub(r'[^a-záéíóúñ\s]', ' ', text)
        
        # Tokenizar por espacios
        tokens = [t.strip() for t in text.split() if t.strip()]
        
        # Remover stopwords y tokens cortos
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        
        return tokens
    """Clase que implementa diferentes algoritmos de similitud de textos."""
    
    @staticmethod
    def levenshtein(texto1: str, texto2: str) -> float:
        """
        Calcula la similitud usando distancia de Levenshtein normalizada.
        
        Args:
            texto1: Primer texto a comparar
            texto2: Segundo texto a comparar
            
        Returns:
            Valor de similitud entre 0 y 1 (1 = textos idénticos)
        """
        if not texto1 or not texto2:
            return 0.0
            
        distancia = Levenshtein.distance(texto1.lower(), texto2.lower())
        max_len = max(len(texto1), len(texto2))
        if max_len == 0:
            return 1.0
            
        # Normalizar para obtener similitud en lugar de distancia
        return 1 - (distancia / max_len)
    
    @staticmethod
    def jaccard(texto1: str, texto2: str) -> float:
        """
        Calcula el coeficiente de similitud de Jaccard.
        
        Args:
            texto1: Primer texto a comparar
            texto2: Segundo texto a comparar
            
        Returns:
            Coeficiente de Jaccard entre 0 y 1
        """
        if not texto1 or not texto2:
            return 0.0
            
        # Convertir a conjuntos de palabras
        set1 = set(texto1.lower().split())
        set2 = set(texto2.lower().split())
        
        # Calcular intersección y unión
        interseccion = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
            
        return interseccion / union
    
    @staticmethod
    def tfidf_coseno(texto1: str, texto2: str) -> float:
        """
        Calcula la similitud coseno usando TF-IDF.
        
        Args:
            texto1: Primer texto a comparar
            texto2: Segundo texto a comparar
            
        Returns:
            Similitud coseno entre 0 y 1
        """
        if not texto1 or not texto2:
            return 0.0
            
        # Crear vectorizador TF-IDF
        vectorizer = TfidfVectorizer()
        
        try:
            # Obtener vectores TF-IDF
            tfidf_matrix = vectorizer.fit_transform([texto1.lower(), texto2.lower()])
            
            # Calcular similitud coseno
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            return float(similarity[0][0])
            
        except Exception:
            return 0.0
    
    @staticmethod
    def ngramas(texto1: str, texto2: str, n: int = 3) -> float:
        """
        Calcula similitud basada en n-gramas.
        
        Args:
            texto1: Primer texto a comparar
            texto2: Segundo texto a comparar
            n: Tamaño de los n-gramas (default=3)
            
        Returns:
            Similitud basada en n-gramas entre 0 y 1
        """
        if not texto1 or not texto2:
            return 0.0
            
        # Generar n-gramas
        def get_ngrams(text: str, n: int) -> set:
            text = text.lower()
            return {text[i:i+n] for i in range(len(text) - n + 1)}
        
        # Obtener conjuntos de n-gramas
        ngrams1 = get_ngrams(texto1, n)
        ngrams2 = get_ngrams(texto2, n)
        
        # Calcular coeficiente de Dice
        interseccion = len(ngrams1.intersection(ngrams2))
        if not ngrams1 and not ngrams2:
            return 1.0
        elif not ngrams1 or not ngrams2:
            return 0.0
            
        return 2.0 * interseccion / (len(ngrams1) + len(ngrams2))
    
    def semantic_embedding(self, texto1: str, texto2: str, n_components: int = 100) -> float:
        """
        Versión simplificada de BERT usando LSA (Latent Semantic Analysis).
        
        Args:
            texto1: Primer texto a comparar
            texto2: Segundo texto a comparar
            n_components: Dimensión del espacio semántico
            
        Returns:
            Similitud semántica entre 0 y 1
        """
        if not texto1 or not texto2:
            return 0.0
            
        try:
            # Vectorización TF-IDF
            vectorizer = TfidfVectorizer(
                tokenizer=self._preprocess_text,
                stop_words=list(self.stop_words)
            )
            
            # Crear matriz TF-IDF
            tfidf_matrix = vectorizer.fit_transform([texto1, texto2])
            
            # Reducción dimensional con LSA
            lsa = TruncatedSVD(n_components=min(n_components, tfidf_matrix.shape[1]-1))
            embeddings = lsa.fit_transform(tfidf_matrix)
            
            # Calcular similitud coseno
            similarity = cosine_similarity(embeddings[0:1], embeddings[1:2])[0][0]
            return float(similarity)
            
        except Exception as e:
            print(f"Error en semantic_embedding: {e}")
            return 0.0
    
    def contextual_similarity(self, texto1: str, texto2: str, window: int = 5) -> float:
        """
        Versión simplificada de Doc2Vec usando ventanas de contexto.
        
        Args:
            texto1: Primer texto a comparar
            texto2: Segundo texto a comparar
            window: Tamaño de la ventana de contexto
            
        Returns:
            Similitud contextual entre 0 y 1
        """
        if not texto1 or not texto2:
            return 0.0
            
        try:
            # Preprocesar textos
            tokens1 = self._preprocess_text(texto1)
            tokens2 = self._preprocess_text(texto2)
            
            # Crear vectores de contexto
            def get_context_vector(tokens: List[str]) -> Dict[str, float]:
                contexts = defaultdict(float)
                for i, token in enumerate(tokens):
                    start = max(0, i - window)
                    end = min(len(tokens), i + window + 1)
                    for j in range(start, end):
                        if j != i:
                            contexts[tokens[j]] += 1.0 / abs(i - j)
                return contexts
            
            # Obtener vectores de contexto
            context1 = get_context_vector(tokens1)
            context2 = get_context_vector(tokens2)
            
            # Calcular similitud de contextos
            common_words = set(context1.keys()) & set(context2.keys())
            if not common_words:
                return 0.0
                
            numerator = sum(context1[w] * context2[w] for w in common_words)
            norm1 = np.sqrt(sum(v*v for v in context1.values()))
            norm2 = np.sqrt(sum(v*v for v in context2.values()))
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return float(numerator / (norm1 * norm2))
            
        except Exception as e:
            print(f"Error en contextual_similarity: {e}")
            return 0.0
    
    def __init_algoritmos(self):
        """Inicializa el diccionario de algoritmos disponibles."""
        self.ALGORITMOS: Dict[str, Callable[[str, str], float]] = {
            'levenshtein': self.__class__.levenshtein,
            'jaccard': self.__class__.jaccard,
            'tfidf_coseno': self.__class__.tfidf_coseno,
            'ngramas': self.__class__.ngramas,
            'semantic': self.semantic_embedding,  # Alternativa a BERT
            'contextual': self.contextual_similarity  # Alternativa a Doc2Vec
        }
        
    def __init__(self):
        """Inicializa recursos necesarios para los algoritmos."""
        # Stop words en español e inglés más comunes
        self.stop_words = {
            'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las',
            'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al', 'lo', 'como',
            'más', 'pero', 'sus', 'le', 'ya', 'o', 'este', 'sí', 'porque',
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her'
        }
        self.__init_algoritmos()
    
    def calcular_similitud(self, 
                      texto1: str, 
                      texto2: str, 
                      algoritmo: str = 'tfidf_coseno') -> float:
        """
        Calcula la similitud entre dos textos usando el algoritmo especificado.
        
        Args:
            texto1: Primer texto a comparar
            texto2: Segundo texto a comparar
            algoritmo: Nombre del algoritmo a usar (default='tfidf_coseno')
            
        Returns:
            Valor de similitud entre 0 y 1
            
        Raises:
            ValueError: Si el algoritmo no está implementado
        """
        if algoritmo not in self.ALGORITMOS:
            raise ValueError(f"Algoritmo '{algoritmo}' no implementado")
            
        return self.ALGORITMOS[algoritmo](texto1, texto2)

def main():
    """Función principal para pruebas."""
    # Textos de ejemplo
    texto1 = "Este es un texto de ejemplo para probar los algoritmos"
    texto2 = "Este es otro texto similar para hacer pruebas"
    
    # Probar todos los algoritmos
    similitud = SimilitudTextos()
    for nombre, _ in SimilitudTextos.ALGORITMOS.items():
        try:
            valor = similitud.calcular_similitud(texto1, texto2, nombre)
            print(f"\nAlgoritmo: {nombre}")
            print(f"Similitud: {valor:.4f}")
        except Exception as e:
            print(f"Error en {nombre}: {e}")

if __name__ == "__main__":
    main()