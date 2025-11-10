#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Grafo semántico no dirigido de términos basado en coocurrencia en abstracts.

Definición:
- Vértices: términos (tokens) seleccionados (top 50 por TF-IDF total).
- Arista no dirigida {i,j} con peso w_ij = número de documentos donde i y j coocurren.

Algoritmos:
- Grados de nodos: deg(i) = número de aristas incidentes (o suma de pesos).
- Componentes conexas mediante DFS propio (recorrido en profundidad).

Exportación: GraphML y JSON legible.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any, Set
from pathlib import Path
import json

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer


def _extraer_abstracts(bib_db: Any) -> List[str]:
    abstracts: List[str] = []
    for entry in getattr(bib_db, "entries", []):
        abs_txt = (entry.get("abstract") or "").strip()
        if abs_txt:
            abstracts.append(abs_txt)
    return abstracts


def seleccionar_top_terminos(abstracts: List[str], k: int = 50) -> Tuple[List[str], List[Set[int]]]:
    """Selecciona top-k términos por suma TF-IDF y devuelve también presencia por documento."""
    if not abstracts:
        return [], []
    vectorizer = TfidfVectorizer(lowercase=True, stop_words=None, max_features=5000)
    X = vectorizer.fit_transform(abstracts)  # shape (n_docs, n_terms)
    vocab = vectorizer.get_feature_names_out()
    tfidf_sums = X.sum(axis=0).A1  # suma por término
    idx_sorted = tfidf_sums.argsort()[::-1]
    idx_top = idx_sorted[: min(k, len(idx_sorted))]
    terms = [vocab[i] for i in idx_top]

    # Presencia por documento para coocurrencia
    term_to_global = {t: i for i, t in enumerate(vocab)}
    presence: List[Set[int]] = []
    for t in terms:
        gidx = term_to_global[t]
        docs = set(X[:, gidx].nonzero()[0].tolist())
        presence.append(docs)
    return terms, presence


def construir_matriz_coocurrencia(presence: List[Set[int]]) -> List[List[int]]:
    n = len(presence)
    M = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            co = len(presence[i] & presence[j])
            M[i][j] = co
            M[i][i] = M[i][i]  # identidad 0
            M[j][i] = co
    return M


def construir_grafo_terminos(terms: List[str], M: List[List[int]], min_coocurrencia: int = 1) -> nx.Graph:
    G = nx.Graph()
    for t in terms:
        G.add_node(t)
    n = len(terms)
    for i in range(n):
        for j in range(i + 1, n):
            w = int(M[i][j])
            if w >= min_coocurrencia:
                G.add_edge(terms[i], terms[j], weight=w)
    return G


def grados(G: nx.Graph, ponderado: bool = False) -> Dict[str, float]:
    if ponderado:
        return {n: float(sum(d.get("weight", 1) for _, _, d in G.edges(n, data=True))) for n in G.nodes}
    return {n: float(G.degree(n)) for n in G.nodes}


def componentes_conexas_dfs(G: nx.Graph) -> List[List[str]]:
    """Componentes conexas por DFS explícito (no usa nx.components)."""
    visitado: Dict[str, bool] = {n: False for n in G.nodes}
    comps: List[List[str]] = []

    def dfs(u: str, comp: List[str]) -> None:
        visitado[u] = True
        comp.append(u)
        for v in G.neighbors(u):
            if not visitado[v]:
                dfs(v, comp)

    for n in G.nodes:
        if not visitado[n]:
            comp: List[str] = []
            dfs(n, comp)
            comps.append(comp)
    return comps


def exportar_graphml(G: nx.Graph, ruta: Path) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, str(ruta))
    return ruta


def exportar_json(G: nx.Graph, ruta: Path) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "directed": False,
        "nodes": [{"id": n, **{k: v for k, v in G.nodes[n].items()}} for n in G.nodes],
        "edges": [
            {"source": u, "target": v, **{k: v2 for k, v2 in G.edges[u, v].items()}} for u, v in G.edges
        ],
    }
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return ruta


def construir_desde_bib(bib_db: Any, k: int = 50, min_coocurrencia: int = 1) -> nx.Graph:
    abstracts = _extraer_abstracts(bib_db)
    terms, presence = seleccionar_top_terminos(abstracts, k=k)
    M = construir_matriz_coocurrencia(presence)
    return construir_grafo_terminos(terms, M, min_coocurrencia=min_coocurrencia)


def main() -> None:
    """Pruebas independientes con textos sintéticos."""
    from collections import namedtuple

    FakeBib = namedtuple("FakeBib", ["entries"])
    bib = FakeBib(
        entries=[
            {"abstract": "graph algorithms shortest path floyd warshall dijkstra"},
            {"abstract": "community detection modularity graph networks"},
            {"abstract": "shortest path algorithms dijkstra networks"},
        ]
    )
    G = construir_desde_bib(bib, k=10, min_coocurrencia=1)
    print(f"Nodos: {G.number_of_nodes()} Aristas: {G.number_of_edges()}")
    print("Grados:", list(grados(G).items())[:5])
    print("Componentes (DFS):", componentes_conexas_dfs(G))


if __name__ == "__main__":
    main()


