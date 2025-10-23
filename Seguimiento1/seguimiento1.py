# -*- coding: utf-8 -*-
"""
Analiza rendimiento de algoritmos de ordenamiento sobre datos enteros
obtenidos del BibTeX unificado (resultado_unificado.bib) y genera:
1) Orden de productos académicos por tipo (año asc, luego título asc).
2) Diagrama de barras (ascendente) de tiempos de 12 algoritmos (PNG si hay matplotlib; CSV siempre).
3) Top 15 autores por apariciones (orden ascendente dentro del top).

PARCHES:
- Import opcional de matplotlib (si no está, no se rompe; solo no genera PNG).
- Sin borrar ninguna funcionalidad previa.
"""

import os
import re
import csv
import time
import math
import random
from pathlib import Path

# Intentar usar matplotlib (si no está, seguimos sin PNG)
try:
    import matplotlib.pyplot as plt  # Solo para exportar PNG del diagrama
    HAVE_MPL = True
    MPL_ERR = ""
except Exception as e:
    HAVE_MPL = False
    MPL_ERR = str(e)

# === RUTA DEL .BIB UNIFICADO ===
BIB_PATH = Path(__file__).parent.parent / "requerimiento1" / "descargas" / "resultado_unificado.bib"

# ---------- Utilidades para cargar datos desde el .bib ----------
def _try_import_bibtexparser():
    try:
        import bibtexparser  # noqa: F401
        return True
    except Exception:
        return False

def _normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def cargar_entries_desde_bib(path: Path):
    """
    Devuelve lista de dicts con claves comunes: ENTRYTYPE, ID, title, year, author, keywords, etc.
    """
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo BibTeX: {path}")

    if _try_import_bibtexparser():
        import bibtexparser
        with open(path, "r", encoding="utf-8") as f:
            db = bibtexparser.load(f)
        # normalizar claves
        entries = []
        for e in db.entries:
            e2 = {k.lower(): v for k, v in e.items()}
            # asegurar title/year/author presentes aunque vacíos
            e2.setdefault("title", e2.get("Title", ""))
            e2.setdefault("year", e2.get("Year", ""))
            e2.setdefault("author", e2.get("Author", ""))
            e2.setdefault("entrytype", e2.get("ENTRYTYPE", e2.get("entrytype", "")))
            e2.setdefault("id", e2.get("ID", e2.get("id", "")))
            entries.append(e2)
        return entries
    else:
        # Parser de respaldo simple
        text = path.read_text(encoding="utf-8", errors="ignore")
        raw = re.split(r"(?m)^@", text)
        entries = []
        for b in raw:
            b = b.strip()
            if not b: continue
            m = re.match(r"(\w+)\s*\{\s*([^,]+)\s*,(.*)", b, re.DOTALL)
            if not m: continue
            entrytype, id_, rest = m.groups()
            fields = {}
            for line in re.split(r",\s*\n", rest):
                kv = line.strip().rstrip(",")
                if "=" not in kv: continue
                k, v = kv.split("=", 1)
                k = k.strip().lower()
                v = v.strip().strip(",").strip()
                if (v.startswith("{") and v.endswith("}")) or (v.startswith('"') and v.endswith('"')):
                    v = v[1:-1]
                fields[k] = v
            e2 = {
                "entrytype": entrytype.lower(),
                "id": id_.strip(),
                "title": fields.get("title", ""),
                "year": fields.get("year", ""),
                "author": fields.get("author", ""),
            }
            # merge resto
            for k, v in fields.items():
                e2.setdefault(k, v)
            entries.append(e2)
        return entries

def cargar_enteros_desde_entries(entries):
    """
    A partir de entries, retorna lista de enteros para ordenar.
    Prioriza 'year'; si no hay, usa len(title).
    """
    enteros = []
    for e in entries:
        y = _normalize_space(str(e.get("year", "")))
        if y.isdigit():
            enteros.append(int(y))
        else:
            t = _normalize_space(e.get("title", ""))
            if t:
                enteros.append(len(t))
    if not enteros:
        raise ValueError("No se encontraron enteros (años o títulos) en el .bib.")
    return enteros

# ---------- Mapeo de tipo de producto ----------
def clasificar_producto(entrytype: str) -> str:
    t = (entrytype or "").lower()
    if t in {"article"}:
        return "Artículo"
    if t in {"inproceedings", "conference", "proceedings"}:
        return "Conferencia"
    if t in {"incollection", "inbook", "bookinbook", "suppbook"}:
        return "Capítulo"
    if t in {"book"}:
        return "Libro"
    # Otros caen en “Artículo” por cercanía, pero puedes cambiarlo
    return "Artículo"

# ---------- Algoritmos de ordenamiento (los 12) ----------
# 1) TimSort
def timsort(arr):  # O(n log n)
    return sorted(arr)
# 2) Comb Sort
def comb_sort(arr):
    a = arr[:]
    gap = len(a)
    shrink = 1.3
    swapped = True
    while gap > 1 or swapped:
        gap = int(gap / shrink) or 1
        swapped = False
        i = 0
        while i + gap < len(a):
            if a[i] > a[i + gap]:
                a[i], a[i + gap] = a[i + gap], a[i]
                swapped = True
            i += 1
    return a
# 3) Selection Sort
def selection_sort(arr):
    a = arr[:]
    n = len(a)
    for i in range(n):
        m = i
        for j in range(i + 1, n):
            if a[j] < a[m]:
                m = j
        a[i], a[m] = a[m], a[i]
    return a
# 4) Tree Sort (BST)
class _BST:
    __slots__ = ("key", "l", "r")
    def __init__(self, key):
        self.key = key; self.l = None; self.r = None
def tree_sort(arr):
    root = None
    for x in arr:
        if root is None:
            root = _BST(x)
        else:
            cur = root
            while True:
                if x <= cur.key:
                    if cur.l: cur = cur.l
                    else: cur.l = _BST(x); break
                else:
                    if cur.r: cur = cur.r
                    else: cur.r = _BST(x); break
    out = []
    def ino(n):
        if not n: return
        ino(n.l); out.append(n.key); ino(n.r)
    ino(root)
    return out
# 5) Pigeonhole Sort
def pigeonhole_sort(arr):
    if not arr: return []
    mn, mx = min(arr), max(arr)
    rango = mx - mn + 1
    if rango > 10_000_000:
        raise MemoryError(f"Rango demasiado grande: {rango}")
    holes = [0] * rango
    for x in arr: holes[x - mn] += 1
    res = []
    for i, c in enumerate(holes):
        if c: res.extend([i + mn] * c)
    return res
# 6) Bucket Sort
def bucket_sort(arr):
    if not arr: return []
    a = arr[:]
    mn, mx = min(a), max(a)
    shift = -mn if mn < 0 else 0
    if shift: a = [x + shift for x in a]; mx += shift
    buckets_count = max(1, int(len(a) ** 0.5))
    buckets = [[] for _ in range(buckets_count)]
    for x in a:
        idx = (x * buckets_count) // (mx + 1 if mx + 1 else 1)
        if idx >= buckets_count: idx = buckets_count - 1
        buckets[idx].append(x)
    res = []
    for b in buckets:
        b.sort()
        res.extend(b)
    if shift: res = [x - shift for x in res]
    return res
# 7) QuickSort
def quicksort(arr):
    a = arr[:]
    def _q(l, r):
        if l >= r: return
        import random
        p = random.randint(l, r)
        a[p], a[r] = a[r], a[p]
        pivot = a[r]
        i = l
        for j in range(l, r):
            if a[j] <= pivot:
                a[i], a[j] = a[j], a[i]
                i += 1
        a[i], a[r] = a[r], a[i]
        _q(l, i - 1); _q(i + 1, r)
    _q(0, len(a) - 1)
    return a
# 8) HeapSort
def heapsort(arr):
    import heapq
    h = arr[:]; heapq.heapify(h)
    return [heapq.heappop(h) for _ in range(len(h))]
# 9) Bitonic Sort
def bitonic_sort(arr):
    if not arr: return []
    a = arr[:]; n = len(a)
    p2 = 1 << (n - 1).bit_length()
    maxv = max(a) if a else 0
    a.extend([maxv] * (p2 - n))
    def comp_swap(i, j, asc):
        if (a[i] > a[j]) == asc:
            a[i], a[j] = a[j], a[i]
    def merge(lo, cnt, asc):
        if cnt > 1:
            k = cnt // 2
            for i in range(lo, lo + cnt - k):
                comp_swap(i, i + k, asc)
            merge(lo, k, asc); merge(lo + k, k, asc)
    def sort_rec(lo, cnt, asc):
        if cnt > 1:
            k = cnt // 2
            sort_rec(lo, k, True); sort_rec(lo + k, k, False); merge(lo, cnt, asc)
    sort_rec(0, len(a), True)
    return a[:n]
# 10) Gnome Sort
def gnome_sort(arr):
    a = arr[:]; i = 0; n = len(a)
    while i < n:
        if i == 0 or a[i] >= a[i-1]: i += 1
        else: a[i], a[i-1] = a[i-1], a[i]; i -= 1
    return a
# 11) Binary Insertion Sort
def binary_insertion_sort(arr):
    a = arr[:]
    for i in range(1, len(a)):
        key = a[i]; lo, hi = 0, i
        while lo < hi:
            mid = (lo + hi) // 2
            if a[mid] <= key: lo = mid + 1
            else: hi = mid
        j = i
        while j > lo:
            a[j] = a[j-1]; j -= 1
        a[lo] = key
    return a
# 12) Radix Sort
def radix_sort(arr):
    if not arr: return []
    a = arr[:]; mn = min(a); shift = -mn if mn < 0 else 0
    if shift: a = [x + shift for x in a]
    mx = max(a); exp = 1
    while mx // exp > 0:
        buckets = [[] for _ in range(10)]
        for x in a: buckets[(x // exp) % 10].append(x)
        a = [x for b in buckets for x in b]; exp *= 10
    if shift: a = [x - shift for x in a]
    return a

# ---------- Ejecutor/medidor ----------
Alg = [
    ("TimSort", "O(n log n)", timsort),
    ("Comb Sort", "~O(n log n)", comb_sort),
    ("Selection Sort", "O(n^2)", selection_sort),
    ("Tree Sort", "O(n log n)*", tree_sort),
    ("Pigeonhole Sort", "O(n + R)", pigeonhole_sort),
    ("BucketSort", "~O(n + k)", bucket_sort),
    ("QuickSort", "O(n log n)*", quicksort),
    ("HeapSort", "O(n log n)", heapsort),
    ("Bitonic Sort", "O(n log^2 n)", bitonic_sort),
    ("Gnome Sort", "O(n^2)", gnome_sort),
    ("Binary Insertion Sort", "O(n^2)", binary_insertion_sort),
    ("RadixSort", "O(d*(n+b))", radix_sort),
]

def medir(nombre, complejidad, fn, datos):
    arr = datos[:]
    t0 = time.perf_counter()
    try:
        res = fn(arr)
        ok = res == sorted(datos)
        ms = (time.perf_counter() - t0) * 1000
        return (nombre, complejidad, len(datos), ms, "OK" if ok else "BAD")
    except MemoryError as me:
        return (nombre, complejidad, len(datos), None, f"SKIP (mem: {me})")
    except RecursionError as re:
        return (nombre, complejidad, len(datos), None, f"SKIP (rec: {re})")
    except Exception as e:
        return (nombre, complejidad, len(datos), None, f"ERR: {e}")

def nice_ms(x):
    return "-" if x is None else f"{x:,.2f}"

# ---------- 1) ORDENAR PRODUCTOS ACADÉMICOS ----------
def exportar_productos_ordenados(entries, out_csv: Path):
    """
    Orden ascendente por: (tipo/grupo), year (num asc), title (asc).
    Exporta CSV con columnas: tipo, year, title, authors, venue, id, doi, url
    """
    filas = []
    for e in entries:
        tipo = clasificar_producto(e.get("entrytype", ""))
        year = _normalize_space(str(e.get("year", "")))
        ynum = int(year) if year.isdigit() else 0
        title = _normalize_space(e.get("title", ""))
        authors = _normalize_space(e.get("author", ""))
        venue = _normalize_space(e.get("journal", e.get("booktitle", "")))
        doi = _normalize_space(e.get("doi", ""))
        url = _normalize_space(e.get("url", ""))
        filas.append((tipo, ynum, title, authors, venue, e.get("id", ""), doi, url))

    # orden: tipo -> año -> título (todo ascendente)
    filas.sort(key=lambda x: (x[0], x[1], x[2].lower()))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tipo", "year", "title", "authors", "venue", "id", "doi", "url"])
        w.writerows(filas)

    print(f"📄 Productos ordenados exportados a: {out_csv}")

# ---------- 2) DIAGRAMA DE BARRAS DE TIEMPOS ----------
def graficar_tiempos(bench_rows, out_png: Path, out_csv: Path):
    """
    bench_rows: lista de tuplas (nombre, complejidad, n, ms, estado)
    Grafica solo las filas con tiempo válido y estado OK (o SKIP/ERR si quieres ver todas).
    Orden ascendente por tiempo.
    """
    validas = [(n, ms) for (n, _, _, ms, _estado) in bench_rows if ms is not None]
    if not validas:
        print("⚠ No hay tiempos válidos para graficar.")
        return
    validas.sort(key=lambda x: x[1])  # ascendente por ms
    nombres = [n for n, _ in validas]
    tiempos = [ms for _, ms in validas]

    # CSV siempre
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algoritmo", "tiempo_ms"])
        for n, ms in validas:
            w.writerow([n, f"{ms:.4f}"])
    print(f"📄 Tiempos exportados a: {out_csv}")

    # Si no hay matplotlib, salimos aquí (sin PNG)
    if not HAVE_MPL:
        print(f"⚠ No se generará el PNG porque matplotlib no está instalado/usable ({MPL_ERR}).")
        return

    # PNG con matplotlib
    plt.figure(figsize=(10, 6))
    plt.bar(nombres, tiempos)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Tiempo (ms)")
    plt.title("Tiempos de algoritmos de ordenamiento (ascendente)")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"🖼  Gráfico guardado en: {out_png}")

# ---------- 3) TOP 15 AUTORES ----------
def top15_autores(entries, out_csv: Path):
    """
    Cuenta apariciones por autor. Separa autores por ' and ', ';', ',' (comunes en BibTeX).
    Selecciona los 15 con más apariciones y los ordena ASCENDENTE (dentro del top).
    """
    from collections import Counter
    c = Counter()
    for e in entries:
        raw = e.get("author", "") or ""
        if not raw: continue
        # separar por conectores típicos
        parts = [raw]
        for sep in (" and ", ";", ","):
            new_parts = []
            for p in parts:
                new_parts.extend([pp.strip() for pp in p.split(sep) if pp.strip()])
            parts = new_parts
        for p in parts:
            if p:
                c[p] += 1

    if not c:
        print("⚠ No se encontraron autores.")
        return

    # top 15 por frecuencia descendente
    top = c.most_common(15)
    # ordenar ascendente por apariciones dentro del top (y alfabético para empates)
    top.sort(key=lambda x: (x[1], x[0].lower()))

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["autor", "apariciones"])
        w.writerows(top)

    print("\n👥 Top 15 autores (ascendente dentro del top):")
    for a, k in top:
        print(f" - {a}: {k}")
    print(f"📄 Top 15 autores exportado a: {out_csv}")

# ---------- Bench de algoritmos ----------
Alg = Alg  # mantener la lista (no tocar)

def main():
    print(f"📥 Cargando BibTex unificado: {BIB_PATH}")
    entries = cargar_entries_desde_bib(BIB_PATH)
    datos = cargar_enteros_desde_entries(entries)

    n = len(datos)
    print(f"🔢 Cantidad de enteros a ordenar: {n}")

    # para estabilidad de testeos
    random.seed(42)
    random.shuffle(datos)

    # rango (útil para Pigeonhole)
    try:
        mn, mx = min(datos), max(datos)
        print(f"📊 Rango de valores: min={mn}, max={mx}, rango={mx-mn+1}")
    except ValueError:
        pass

    # Medición de algoritmos (se conserva lo existente)
    filas = []
    for nombre, comp, fn in Alg:
        r = medir(nombre, comp, fn, datos)
        filas.append(r)

    # Tabla en consola (se conserva)
    ancho = (28, 14, 10, 12, 14)
    print("\n" + "Tabla 1. Análisis de datos enteros".center(sum(ancho) + 8))
    print("-" * (sum(ancho) + 8))
    print(f"{'Método de ordenamiento':<{ancho[0]}} | {'Complejidad':<{ancho[1]}} | {'Tamaño':<{ancho[2]}} | {'Tiempo (ms)':<{ancho[3]}} | {'Check':<{ancho[4]}}")
    print("-" * (sum(ancho) + 8))
    for (nombre, comp, tam, ms, estado) in filas:
        print(f"{nombre:<{ancho[0]}} | {comp:<{ancho[1]}} | {tam:<{ancho[2]}} | {nice_ms(ms):<{ancho[3]}} | {estado:<{ancho[4]}}")
    print("-" * (sum(ancho) + 8))

    # === NUEVAS SALIDAS ===
    out_dir = BIB_PATH.parent  # misma carpeta del .bib unificado
    exportar_productos_ordenados(entries, out_dir / "productos_ordenados.csv")
    graficar_tiempos(filas, out_dir / "algoritmos_tiempos.png", out_dir / "algoritmos_tiempos.csv")
    top15_autores(entries, out_dir / "autores_top15.csv")

if __name__ == "__main__":
    main()
