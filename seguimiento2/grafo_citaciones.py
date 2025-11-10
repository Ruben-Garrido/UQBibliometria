#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Construcción y análisis de una red de citaciones a partir de referencias BibTeX.

Definiciones y notas:
- Sea G = (V, E) un grafo dirigido donde V son artículos y E arcos de citación.
- Inferimos una citación u->v si el título de u es similar al título de v
  (similitud coseno TF-IDF > umbral) y year(u) > year(v). El peso w(u,v) = 1 - sim.

Algoritmos implementados:
- Caminos mínimos:
  - Dijkstra: para fuente única. Complejidad O((|V|+|E|) log |V|)
  - Floyd-Warshall: para todos los pares. Complejidad O(|V|^3)
- Componentes fuertemente conexas (SCC) mediante Tarjan:
  - Tarjan realiza un DFS manteniendo índices y lowlinks; produce partición de V en SCCs.

Exportación: GraphML y JSON legible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import json

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Articulo:
    """Representa un artículo mínimo para el grafo.

    Atributos:
        id: Identificador único (BibTeX ID)
        titulo: Título del artículo
        anio: Año de publicación (si no disponible, None)
        abstract: Resumen (opcional)
    """

    id: str
    titulo: str
    anio: Optional[int]
    abstract: str = ""


def _extraer_articulos_desde_bib(bib_db: Any) -> List[Articulo]:
    articulos: List[Articulo] = []
    for entry in getattr(bib_db, "entries", []):
        id_art = entry.get("ID") or entry.get("id") or ""
        titulo = entry.get("title", "").strip()
        abstract = entry.get("abstract", "").strip()
        anio_raw = entry.get("year") or entry.get("date")
        anio: Optional[int] = None
        if anio_raw:
            try:
                anio = int(str(anio_raw)[:4])
            except Exception:
                anio = None
        if id_art and titulo:
            articulos.append(Articulo(id=id_art, titulo=titulo, anio=anio, abstract=abstract))
    return articulos


def inferir_citaciones_por_titulo(articulos: List[Articulo], umbral: float = 0.85) -> List[Tuple[str, str, float]]:
    """Infere arcos de citación usando similitud de títulos con TF-IDF y coseno.

    Dado el vector TF-IDF de títulos, definimos sim(i,j) = coseno(titulo_i, titulo_j).
    Se crea un arco dirigido i -> j si sim(i,j) > umbral y anio(i) > anio(j). El peso
    será 1 - sim(i,j) para que represente costo en caminos mínimos.

    Args:
        articulos: lista de artículos.
        umbral: valor en (0,1]. Valores altos dan coincidencias estrictas.

    Returns:
        Lista de tuplas (source_id, target_id, peso)
    """
    if not articulos:
        return []

    titulos = [a.titulo for a in articulos]
    vectorizer = TfidfVectorizer(lowercase=True, stop_words=None)
    X = vectorizer.fit_transform(titulos)
    sim = cosine_similarity(X)

    id_por_idx = [a.id for a in articulos]
    anio_por_idx = [a.anio for a in articulos]

    edges: List[Tuple[str, str, float]] = []
    n = len(articulos)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            s = float(sim[i, j])
            if s > umbral:
                ai, aj = anio_por_idx[i], anio_por_idx[j]
                if ai is not None and aj is not None and ai > aj:
                    # i (más nuevo) cita a j (más antiguo)
                    peso = max(1.0 - s, 1e-6)
                    edges.append((id_por_idx[i], id_por_idx[j], peso))
    return edges


def construir_grafo_citaciones(articulos: List[Articulo], edges: List[Tuple[str, str, float]]) -> nx.DiGraph:
    """Construye un DiGraph de citaciones con atributos útiles en nodos y aristas."""
    G = nx.DiGraph()
    for a in articulos:
        G.add_node(a.id, titulo=a.titulo, anio=a.anio, abstract=a.abstract)
    for u, v, w in edges:
        if u in G and v in G:
            G.add_edge(u, v, weight=float(w), similarity=float(1.0 - w))
    return G


def dijkstra(G: nx.DiGraph, fuente: str) -> Dict[str, float]:
    """Caminos mínimos de fuente única (Dijkstra) usando pesos 'weight'."""
    return nx.single_source_dijkstra_path_length(G, fuente, weight="weight")


def floyd_warshall(G: nx.DiGraph) -> Dict[str, Dict[str, float]]:
    """Caminos mínimos para todos los pares (Floyd-Warshall) usando pesos 'weight'."""
    return nx.floyd_warshall(G, weight="weight")


def tarjan_scc(G: nx.DiGraph) -> List[List[str]]:
    """Implementación de Tarjan para SCC.

    Mantiene un índice incremental y para cada nodo u guarda: index[u] y lowlink[u].
    Durante DFS se apilan nodos; cuando lowlink[u] == index[u], se extrae una SCC.
    """
    index = 0
    stack: List[str] = []
    on_stack: Dict[str, bool] = {}
    indices: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    sccs: List[List[str]] = []

    def strongconnect(u: str) -> None:
        nonlocal index
        indices[u] = index
        lowlink[u] = index
        index += 1
        stack.append(u)
        on_stack[u] = True

        for v in G.successors(u):
            if v not in indices:
                strongconnect(v)
                lowlink[u] = min(lowlink[u], lowlink[v])
            elif on_stack.get(v, False):
                lowlink[u] = min(lowlink[u], indices[v])

        if lowlink[u] == indices[u]:
            # raíz de una SCC
            comp: List[str] = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                comp.append(w)
                if w == u:
                    break
            sccs.append(comp)

    for u in G.nodes:
        if u not in indices:
            strongconnect(u)

    return sccs


def exportar_graphml(G: nx.DiGraph, ruta: Path) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, str(ruta))
    return ruta


def exportar_json(G: nx.DiGraph, ruta: Path) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "directed": True,
        "nodes": [{"id": n, **{k: v for k, v in G.nodes[n].items()}} for n in G.nodes],
        "edges": [
            {"source": u, "target": v, **{k: v2 for k, v2 in G.edges[u, v].items()}} for u, v in G.edges
        ],
    }
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return ruta


def construir_desde_bib(bib_db: Any, umbral: float = 0.85) -> nx.DiGraph:
    articulos = _extraer_articulos_desde_bib(bib_db)
    edges = inferir_citaciones_por_titulo(articulos, umbral=umbral)
    return construir_grafo_citaciones(articulos, edges)


def main() -> None:
    """Pruebas independientes del módulo con datos mínimos sintéticos."""
    from collections import namedtuple

    FakeBib = namedtuple("FakeBib", ["entries"])
    bib = FakeBib(
        entries=[
            {"ID": "A", "title": "Graph algorithms for networks", "year": "2020"},
            {"ID": "B", "title": "Algorithms on graphs and networks", "year": "2018"},
            {"ID": "C", "title": "Neural networks in vision", "year": "2017"},
        ]
    )
    G = construir_desde_bib(bib, umbral=0.5)
    print(f"Nodos: {G.number_of_nodes()} Aristas: {G.number_of_edges()}")
    # Dijkstra desde A
    if "A" in G:
        print("Dijkstra desde A:", dijkstra(G, "A"))
    # Floyd
    dist_fw = floyd_warshall(G)
    print("FW pares:", {u: dict(list(d.items())[:2]) for u, d in dist_fw.items()})
    # Tarjan
    print("SCCs:", tarjan_scc(G))


if __name__ == "__main__":
    main()


