#!/usr/bin/env python3
"""Tabla de fuerza vs intervalos de aceleración (Lab 2, El Elevador).
Uso: python analisis_elevador.py [config.yaml]"""

import csv
import re
import sys
import unicodedata

import numpy as np
import pandas as pd
import yaml


def cargar_config(ruta):
    with open(ruta, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("k", 10)
    cfg.setdefault("paso", cfg["k"])
    cfg.setdefault("n", 10)
    cfg.setdefault("bins", "ancho")
    cfg.setdefault("masa", None)
    cfg.setdefault("g", 9.81)
    cfg.setdefault("series", None)
    cfg.setdefault("invertir_velocidad", False)
    cfg.setdefault("salida", "tabla_intervalos_aceleracion.csv")
    return cfg


def normalizar(s):
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn").lower()


def a_numero(x):
    s = str(x).strip().replace("\u00a0", "").replace(" ", "").replace("\u2212", "-")
    if s == "":
        return np.nan
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def tipo_columna(nombre):
    n = normalizar(nombre)
    if "tiempo" in n or "time" in n:
        return "t"
    if "fuerza" in n or "force" in n:
        return "F"
    if "velocidad" in n or "velocity" in n:
        return "v"
    return None


def leer_csv(ruta):
    with open(ruta, "rb") as f:
        crudo = f.read()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            lineas = crudo.decode(enc).splitlines()
            break
        except UnicodeDecodeError:
            continue

    # La cabecera es la primera línea con "tiempo" (salta títulos de Numbers)
    idx = next(i for i, l in enumerate(lineas) if "tiempo" in normalizar(l) or "time" in normalizar(l))
    sep = max([";", "\t", ","], key=lineas[idx].count)
    filas = list(csv.reader(lineas[idx:], delimiter=sep))
    cab = [h.strip() for h in filas[0]]
    cuerpo = [(f + [""] * len(cab))[:len(cab)] for f in filas[1:] if any(c.strip() for c in f)]
    datos = pd.DataFrame(cuerpo, columns=cab)

    # Agrupa columnas por número de serie
    grupos = {}
    for col in cab:
        tipo = tipo_columna(col)
        if tipo:
            m = re.search(r"(?:serie|run)\s*(?:n[º°o.]?\s*|#\s*)?(\d+)", col, re.I)
            grupos.setdefault(int(m.group(1)) if m else 0, {})[tipo] = col

    series = {}
    for s, cols in grupos.items():
        if "t" not in cols and "t" in grupos.get(0, {}):
            cols["t"] = grupos[0]["t"]
        if all(k in cols for k in "tFv"):
            df = pd.DataFrame({k: datos[cols[k]].map(a_numero) for k in "tFv"})
            series[s] = df.dropna().sort_values("t").reset_index(drop=True)
    return series


def ventanas(df, k, paso):
    t, F, v = df["t"].to_numpy(), df["F"].to_numpy(), df["v"].to_numpy()
    dt = np.median(np.diff(t))
    filas = []
    for i in range(0, len(t) - k + 1, paso):
        tt, vv, FF = t[i:i + k], v[i:i + k], F[i:i + k]
        if np.max(np.diff(tt)) > 1.5 * dt:  # descarta ventanas con datos faltantes
            continue
        x = tt - tt.mean()
        a = np.sum(x * (vv - vv.mean())) / np.sum(x ** 2)  # pendiente de v(t)
        filas.append({"a": a, "F": FF.mean()})
    return pd.DataFrame(filas)


def tabla(w, cfg):
    if cfg["bins"] == "cuantil":
        w["intervalo"] = pd.qcut(w["a"], q=cfg["n"], duplicates="drop")
    else:
        w["intervalo"] = pd.cut(w["a"], bins=cfg["n"])

    filas = []
    for itv, g in w.groupby("intervalo", observed=True):
        nv = len(g)
        f = {
            "a_desde": itv.left, "a_hasta": itv.right, "n_ventanas": nv,
            "a_prom": g["a"].mean(), "a_mediana": g["a"].median(),
            "a_std": g["a"].std(ddof=1) if nv > 1 else np.nan,
            "F_prom": g["F"].mean(), "F_mediana": g["F"].median(),
            "F_std": g["F"].std(ddof=1) if nv > 1 else np.nan,
        }
        f["F_err_prom"] = f["F_std"] / np.sqrt(nv) if nv > 1 else np.nan
        if cfg["masa"]:
            f["N_teorica"] = cfg["masa"] * (cfg["g"] + f["a_prom"])
            f["dif_%"] = 100 * (f["F_prom"] - f["N_teorica"]) / f["N_teorica"]
        filas.append(f)
    return pd.DataFrame(filas)


def main():
    cfg = cargar_config(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
    series = leer_csv(cfg["archivo"])
    if cfg["series"]:
        series = {s: df for s, df in series.items() if s in cfg["series"]}

    todas = []
    for df in series.values():
        if cfg["invertir_velocidad"]:
            df["v"] = -df["v"]
        todas.append(ventanas(df, cfg["k"], cfg["paso"]))
    w = pd.concat(todas, ignore_index=True)

    res = tabla(w, cfg)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    res.to_csv(cfg["salida"], sep=";", decimal=",", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
