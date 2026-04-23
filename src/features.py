"""
src/features.py
---------------
Feature engineering para o projeto de previsão de incêndios.
Inclui: Índice de Angstrom, FMA, features temporais e geoespaciais.
"""

import pandas as pd
import numpy as np


# ── Índice de Angstrom ──────────────────────────────────────────────────────
def indice_angstrom(umidade: pd.Series, temperatura: pd.Series) -> pd.Series:
    """
    Calcula o Índice de Angstrom (IA) para risco de fogo.

    IA = H/20 + (27 - T) / 10
    onde H = umidade relativa (%) e T = temperatura (°C)

    Interpretação:
        IA < 2.0  → risco muito alto
        2.0–3.9   → risco alto
        4.0–5.9   → risco moderado
        ≥ 6.0     → sem risco
    """
    return (umidade / 20.0) + ((27.0 - temperatura) / 10.0)


def classificar_angstrom(ia: pd.Series) -> pd.Series:
    """Classifica o risco pelo Índice de Angstrom."""
    bins   = [-np.inf, 2.0, 4.0, 6.0, np.inf]
    labels = ["Muito Alto", "Alto", "Moderado", "Sem Risco"]
    return pd.cut(ia, bins=bins, labels=labels)


# ── Fórmula de Monte Alegre (FMA) ──────────────────────────────────────────
def calcular_fma(
    series_umidade: pd.Series,
    series_precip: pd.Series,
    limiar_mm: float = 13.0
) -> list:
    """
    Calcula a Fórmula de Monte Alegre (FMA) acumulativa.

    FMA é reiniciada quando precipitação ≥ limiar_mm.
    Deve ser aplicada por série temporal de uma localidade específica.

    Args:
        series_umidade : Série de umidade relativa (%)
        series_precip  : Série de precipitação diária (mm)
        limiar_mm      : Precipitação mínima para reiniciar acumulação

    Returns:
        Lista com valores FMA para cada dia
    """
    fma_vals = []
    acum = 0.0
    for h, p in zip(series_umidade, series_precip):
        if pd.isna(h) or pd.isna(p):
            fma_vals.append(np.nan)
            continue
        if p >= limiar_mm:
            acum = 0.0
        else:
            acum += h / (13.0 - h + 1e-6)
        fma_vals.append(acum)
    return fma_vals


def classificar_fma(fma: pd.Series) -> pd.Series:
    """Classifica o risco pelo FMA."""
    bins   = [-np.inf, 1.0, 3.0, 8.0, np.inf]
    labels = ["Nulo", "Pequeno", "Médio", "Alto"]
    return pd.cut(fma, bins=bins, labels=labels)


# ── Features Temporais ──────────────────────────────────────────────────────
ESTACAO_MAP = {
    12: "Verão",   1: "Verão",    2: "Verão",
     3: "Outono",  4: "Outono",   5: "Outono",
     6: "Inverno", 7: "Inverno",  8: "Inverno",
     9: "Primavera", 10: "Primavera", 11: "Primavera",
}

def add_temporal_features(df: pd.DataFrame, col_datetime: str = "data_hora") -> pd.DataFrame:
    """
    Adiciona features temporais derivadas da coluna de data/hora.

    Cria: mes, estacao, dia_do_ano, hora, dia_semana
    """
    df = df.copy()
    dt = pd.to_datetime(df[col_datetime])
    df["mes"]         = dt.dt.month
    df["estacao"]     = df["mes"].map(ESTACAO_MAP)
    df["dia_do_ano"]  = dt.dt.dayofyear
    df["hora"]        = dt.dt.hour
    df["dia_semana"]  = dt.dt.dayofweek  # 0=segunda, 6=domingo
    df["ano"]         = dt.dt.year
    return df


# ── Features Geoespaciais ───────────────────────────────────────────────────
def add_grid_cell(
    df: pd.DataFrame,
    resolucao: float = 0.5,
    lat_col: str = "latitude",
    lon_col: str = "longitude"
) -> pd.DataFrame:
    """
    Cria célula de grade geoespacial (lat_grid, lon_grid) com a resolução
    especificada em graus. Padrão: 0.5° × 0.5° (~55km).
    """
    df = df.copy()
    df["lat_grid"] = (df[lat_col] / resolucao).round(0) * resolucao
    df["lon_grid"] = (df[lon_col] / resolucao).round(0) * resolucao
    df["cell_id"]  = df["lat_grid"].astype(str) + "_" + df["lon_grid"].astype(str)
    return df


# ── Encoding de Bioma ───────────────────────────────────────────────────────
def encode_bioma(df: pd.DataFrame, col: str = "bioma") -> pd.DataFrame:
    """Aplica one-hot encoding na coluna de bioma."""
    return pd.get_dummies(df, columns=[col], prefix="bioma", dtype=int)


# ── Classificação de Severidade FRP ────────────────────────────────────────
def classificar_frp(frp: pd.Series) -> pd.Series:
    """
    Classifica o FRP em categorias de severidade.

    Baseado em Wooster et al. (2005) e dados do INPE:
        < 10 MW  → Baixo
        10–50    → Médio
        50–200   → Alto
        > 200    → Extremo
    """
    bins   = [-np.inf, 10, 50, 200, np.inf]
    labels = ["Baixo", "Médio", "Alto", "Extremo"]
    return pd.cut(frp, bins=bins, labels=labels)
