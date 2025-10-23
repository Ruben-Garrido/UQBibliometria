# -*- coding: utf-8 -*-
"""
Une y deduplica BibTeX de ACM + ScienceDirect en un solo archivo.
Criterio de duplicado: título (normalizado).
Fusión de campos para preservar la mayor cantidad de información.
"""

import os
import re
import sys
import unicodedata
from pathlib import Path

# Ruta relativa a la carpeta de descargas
BASE_DIR = Path(__file__).parent / "descargas"
# Asegurarse de que las carpetas existan
BASE_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "acm").mkdir(exist_ok=True)
(BASE_DIR / "ScienceDirect").mkdir(exist_ok=True)

# Corregir las rutas para buscar en las subcarpetas correctas
ACM_BIB = BASE_DIR / "acm" / "acmCompleto.bib"
SCIENCE_BIB = BASE_DIR / "ScienceDirect" / "sciencedirectCompleto.bib"
SALIDA_BIB = BASE_DIR / "resultado_unificado.bib"

# -------- Utilidades de normalización / split --------

PUNCT_PATTERN = re.compile(r"[^\w\s]", re.UNICODE)

def strip_braces(s: str) -> str:
    return s.replace("{", "").replace("}", "")

def normalize_text(s: str) -> str:
    s = strip_braces(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = PUNCT_PATTERN.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def smart_union_list(values, sep_candidates=(" and ", ";", ",")):
    """
    Une listas representadas como string usando separadores típicos en BibTeX.
    Devuelve string con separador '; ' sin duplicados, preservando orden de aparición.
    """
    seen = set()
    ordered = []
    for s in values:
        if s is None:
            continue
        txt = s.strip()
        if not txt:
            continue
        parts = [txt]
        # romper por el primer separador que funcione "bien"
        # intentaremos todos para capturar casos mixtos
        for sep in sep_candidates:
            new_parts = []
            for p in parts:
                new_parts.extend([pp.strip() for pp in p.split(sep) if pp.strip()])
            parts = new_parts
        for p in parts:
            key = normalize_text(p)
            if key not in seen:
                seen.add(key)
                ordered.append(p)
    return "; ".join(ordered)

def merge_scalar(a: str, b: str) -> str:
    """
    Para campos escalares: si uno contiene al otro, usa el más largo.
    Si son distintos y no contenidos, concatena con ' | ' sin duplicar.
    """
    if not a: return b
    if not b: return a
    if a == b: return a
    if a in b: return b
    if b in a: return a
    # evitar concatenar si normalizados son iguales
    if normalize_text(a) == normalize_text(b):
        return a if len(a) >= len(b) else b
    return f"{a} | {b}"

# -------- Intentar usar bibtexparser, con fallback sencillo --------

def _load_with_bibtexparser(path: Path):
    import bibtexparser
    with open(path, "r", encoding="utf-8") as f:
        db = bibtexparser.load(f)
    # Representar cada entrada como dict plano
    return db

def _dump_with_bibtexparser(entries, salida: Path):
    import bibtexparser
    from bibtexparser.bwriter import BibTexWriter
    from bibtexparser.bibdatabase import BibDatabase

    db = BibDatabase()
    db.entries = entries
    writer = BibTexWriter()
    writer.indent = "  "
    writer.order_entries_by = ("ID",)
    with open(salida, "w", encoding="utf-8") as f:
        f.write(writer.write(db))

def _simple_bib_parse(text: str):
    """
    Parser simple de respaldo (no cubre 100% de casos, pero funciona para la mayoría).
    Devuelve lista de dicts con claves: ENTRYTYPE, ID y campos.
    """
    entries = []
    # separar por entradas @
    for block in re.split(r"(?m)^@", text):
        block = block.strip()
        if not block:
            continue
        # ejemplo: ARTICLE{id,
        m = re.match(r"(\w+)\s*\{\s*([^,]+)\s*,(.*)\}\s*$", block, re.DOTALL)
        if not m:
            # intentar otra forma hasta primera llave
            m2 = re.match(r"(\w+)\s*\{\s*([^,]+)\s*,(.*)", block, re.DOTALL)
            if not m2:
                continue
            entrytype, id_, rest = m2.groups()
        else:
            entrytype, id_, rest = m.groups()
        fields = {}
        # partir por líneas field = {value} o field = "value"
        for line in re.split(r",\s*\n", rest):
            kv = line.strip().rstrip(",")
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            k = k.strip()
            v = v.strip().strip(",")
            # quitar braces o comillas envolventes
            if (v.startswith("{") and v.endswith("}")) or (v.startswith('"') and v.endswith('"')):
                v = v[1:-1].strip()
            if k:
                fields[k.lower()] = v
        entry = {"ENTRYTYPE": entrytype.lower(), "ID": id_.strip()}
        entry.update(fields)
        entries.append(entry)
    return entries

def _simple_bib_dump(entries, salida: Path):
    def fmt_field(k, v):
        return f"  {k} = {{{v}}}"
    lines = []
    for e in entries:
        et = e.get("ENTRYTYPE", "article")
        id_ = e.get("ID", "noid")
        # ordenar campos dejando ENTRYTYPE e ID fuera
        body_fields = [(k, v) for k, v in e.items() if k not in ("ENTRYTYPE", "ID")]
        body = ",\n".join(fmt_field(k, v) for k, v in body_fields if v)
        block = f"@{et}{{{id_},\n{body}\n}}\n"
        lines.append(block)
    salida.write_text("".join(lines), encoding="utf-8")

def cargar_bib(path: Path):
    if not path.exists():
        print(f"⚠ No existe: {path}")
        return []
    try:
        db = _load_with_bibtexparser(path)
        return db.entries  # type: ignore[attr-defined]
    except Exception as e:
        print(f"ℹ bibtexparser no disponible o falló ({e}). Usando parser simple.")
        text = path.read_text(encoding="utf-8")
        return _simple_bib_parse(text)

def guardar_bib(entries, salida: Path):
    try:
        _dump_with_bibtexparser(entries, salida)
    except Exception as e:
        print(f"ℹ No pude usar bibtexparser para escribir ({e}). Usando volcado simple.")
        _simple_bib_dump(entries, salida)

# -------- Lógica de fusión / deduplicación --------

LIST_FIELDS = {
    "author": (" and ", ";", ","),
    "keywords": (";", ","),
}
UNION_FIELDS = {"author", "keywords", "doi", "url"}  # doi/url también se unen sin repetir

def normalized_title(entry: dict) -> str:
    title = entry.get("title", "") or entry.get("Title", "")
    return normalize_text(title)

def merge_entries(e1: dict, e2: dict) -> dict:
    merged = dict(e1)  # base
    # Alinear ENTRYTYPE
    et1 = e1.get("ENTRYTYPE", "")
    et2 = e2.get("ENTRYTYPE", "")
    if et1 and et2 and et1 != et2:
        merged.setdefault("entrytype_alt", et2)
    elif not et1 and et2:
        merged["ENTRYTYPE"] = et2

    # IDs alternativos (para rastreabilidad)
    aliases = []
    if "ID" in e1: aliases.append(e1["ID"])
    if "ID" in e2 and e2["ID"] != e1.get("ID"): aliases.append(e2["ID"])
    if aliases:
        merged["aliases"] = smart_union_list(["; ".join(aliases)], sep_candidates=(";",))

    # Fusionar campos
    keys = set(e1.keys()) | set(e2.keys())
    for k in keys:
        if k in ("ENTRYTYPE", "ID"):  # ya tratados
            continue
        v1 = e1.get(k, "")
        v2 = e2.get(k, "")
        if not v1 and not v2:
            continue

        lk = k.lower()
        # Campos de lista → unión
        if lk in LIST_FIELDS:
            merged[k] = smart_union_list([v1, v2], sep_candidates=LIST_FIELDS[lk])
            continue

        # doi / url → unión sin repetir
        if lk in {"doi", "url"}:
            merged[k] = smart_union_list([v1, v2], sep_candidates=(";", ",", " "))
            continue

        # abstract → concatenar sin repetir
        if lk == "abstract":
            if not v1: merged[k] = v2
            elif not v2: merged[k] = v1
            else:
                merged[k] = merge_scalar(v1, v2)
            continue

        # title → mantener el más informativo (pero no cambiamos clave de dedup)
        if lk == "title":
            merged[k] = merge_scalar(v1, v2)
            continue

        # Resto de campos escalares → merge sin perder información
        merged[k] = merge_scalar(v1, v2)

    return merged

def main():
    print(f"📥 Leyendo:\n - {ACM_BIB}\n - {SCIENCE_BIB}")
    acm_entries = cargar_bib(ACM_BIB)
    sd_entries = cargar_bib(SCIENCE_BIB)

    all_entries = []
    all_entries.extend(acm_entries)
    all_entries.extend(sd_entries)

    if not all_entries:
        print("❌ No se encontraron entradas en los .bib de origen.")
        sys.exit(1)

    # Deduplicar por título normalizado
    by_title = {}
    for e in all_entries:
        tkey = normalized_title(e)
        if not tkey:
            # sin título, hacemos clave por ID (si existe) para no perder el registro
            tkey = f"__notitle__::{e.get('ID','noid')}"
        if tkey in by_title:
            by_title[tkey] = merge_entries(by_title[tkey], e)
        else:
            by_title[tkey] = e

    # Reasignar ID seguro (si hubo fusiones)
    final_entries = []
    for idx, (tkey, e) in enumerate(by_title.items(), start=1):
        if not e.get("ID"):
            # genera un ID estable desde el título normalizado (recortado) + índice
            base = re.sub(r"\s+", "_", tkey)[:40].strip("_")
            e["ID"] = f"{base or 'entry'}_{idx}"
        final_entries.append(e)

    print(f"🧮 Entradas originales: {len(all_entries)}")
    print(f"✅ Entradas tras deduplicar: {len(final_entries)}")

    guardar_bib(final_entries, SALIDA_BIB)
    print(f"💾 Archivo unificado escrito en: {SALIDA_BIB}")

if __name__ == "__main__":
    main()
