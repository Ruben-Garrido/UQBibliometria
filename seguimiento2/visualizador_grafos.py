#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualización interactiva de grafos de citaciones y términos usando Plotly.

Layouts:
- Citaciones (DiGraph): layout jerárquico por año (niveles en eje Y decreciente).
- Términos (Graph): layout por comunidades (greedy modularity) coloreando por comunidad.

Exportación a PNG requiere 'kaleido' instalado.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple
import random

import networkx as nx
import plotly.graph_objects as go


def _random_color() -> str:
    return f"hsl({random.randint(0, 359)},70%,50%)"


def layout_jerarquico_por_anio(G: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
    # Agrupar por año (None al final)
    niveles: Dict[int, list] = {}
    sin_anio = []
    for n, d in G.nodes(data=True):
        anio = d.get("anio")
        if isinstance(anio, int):
            niveles.setdefault(anio, []).append(n)
        else:
            sin_anio.append(n)
    niveles_ordenados = sorted(niveles.items(), key=lambda kv: -kv[0])

    pos: Dict[str, Tuple[float, float]] = {}
    y_step = 1.0
    y = 0.0
    for _, nodes in niveles_ordenados:
        x_step = 1.0 / max(1, len(nodes))
        for i, n in enumerate(nodes):
            pos[n] = (i * x_step, y)
        y -= y_step
    # sin año al final
    if sin_anio:
        x_step = 1.0 / max(1, len(sin_anio))
        for i, n in enumerate(sin_anio):
            pos[n] = (i * x_step, y)
    return pos


def layout_comunidades(G: nx.Graph) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, int]]:
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comunidades = list(greedy_modularity_communities(G))
    except Exception:
        comunidades = [set(G.nodes)]
    # asignar comunidad por nodo
    nodo_comunidad: Dict[str, int] = {}
    for idx, c in enumerate(comunidades):
        for n in c:
            nodo_comunidad[n] = idx
    # posiciones con spring layout
    pos = nx.spring_layout(G, seed=42, k=None)
    return pos, nodo_comunidad


def fig_digraph(G: nx.DiGraph) -> go.Figure:
    pos = layout_jerarquico_por_anio(G)
    # aristas
    edge_x = []
    edge_y = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#888"), hoverinfo="none")

    # nodos
    node_x = []
    node_y = []
    texts = []
    for n, d in G.nodes(data=True):
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        texts.append(d.get("titulo", n))
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=texts,
        marker=dict(showscale=False, color="#1f77b4", size=12, line=dict(width=1, color="#333")),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
    )
    return fig


def fig_graph(G: nx.Graph) -> go.Figure:
    pos, nodo_comunidad = layout_comunidades(G)
    colores = {}
    # aristas
    edge_x = []
    edge_y = []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#aaa"), hoverinfo="none")

    # nodos por comunidad
    data_traces = [edge_trace]
    for n, (x, y) in pos.items():
        cid = nodo_comunidad.get(n, 0)
        if cid not in colores:
            colores[cid] = _random_color()
        color = colores[cid]
        data_traces.append(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers",
                hoverinfo="text",
                text=[n],
                marker=dict(size=10, color=color, line=dict(width=1, color="#333")),
                name=f"comunidad {cid}",
            )
        )

    fig = go.Figure(data=data_traces)
    fig.update_layout(
        showlegend=True,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
    )
    return fig


def exportar_png(fig: go.Figure, ruta: str) -> str:
    fig.write_image(ruta, format="png", scale=2)
    return ruta


def main() -> None:
    import networkx as nx

    Gd = nx.DiGraph()
    Gd.add_node("A", anio=2020, titulo="A")
    Gd.add_node("B", anio=2019, titulo="B")
    Gd.add_edge("A", "B")
    fig1 = fig_digraph(Gd)
    fig1.show()

    G = nx.Graph()
    G.add_edges_from([("t1", "t2"), ("t2", "t3"), ("t4", "t5")])
    fig2 = fig_graph(G)
    fig2.show()


if __name__ == "__main__":
    main()


