"""
src/clustering.py
-----------------
Clusterização geoespacial de focos de incêndio usando DBSCAN (haversine).
Reduz registros agrupando focos dentro de um raio geográfico.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN


def clusterizar_focos(
    df: pd.DataFrame,
    raio_km: float = 100.0,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    frp_col: str = "frp",
) -> pd.DataFrame:
    """
    Agrupa focos de incêndio dentro de um raio de até `raio_km` km
    usando DBSCAN com métrica haversine.

    O ponto representante de cada cluster é o de maior FRP
    (Fire Radiative Power — mais intenso).

    Args:
        df      : DataFrame com colunas de lat/lon
        raio_km : Raio máximo para agrupar focos (padrão: 100km)
        lat_col : Nome da coluna de latitude
        lon_col : Nome da coluna de longitude
        frp_col : Nome da coluna de FRP para selecionar representante

    Returns:
        DataFrame reduzido com um ponto por cluster
    """
    df = df.copy().dropna(subset=[lat_col, lon_col])

    coords  = np.radians(df[[lat_col, lon_col]].values)
    epsilon = raio_km / 6371.0  # converter km → radianos

    db = DBSCAN(
        eps=epsilon,
        min_samples=1,
        algorithm="ball_tree",
        metric="haversine",
        n_jobs=-1,
    ).fit(coords)

    df["cluster_id"] = db.labels_

    # Representante: ponto de maior FRP no cluster
    if frp_col in df.columns:
        df_agrupado = (
            df.sort_values(frp_col, ascending=False, na_position="last")
              .groupby("cluster_id")
              .first()
              .reset_index(drop=True)
        )
    else:
        df_agrupado = (
            df.groupby("cluster_id")
              .first()
              .reset_index(drop=True)
        )

    return df_agrupado


def clusterizar_por_janela(
    df: pd.DataFrame,
    raio_km: float = 100.0,
    janela: str = "W",
    data_col: str = "data_hora",
) -> pd.DataFrame:
    """
    Clusteriza por janela temporal (dia/semana) para garantir
    coerência temporal dos agrupamentos.

    Args:
        df      : DataFrame com coluna de data e lat/lon
        raio_km : Raio de agrupamento em km
        janela  : Frequência pandas ('D'=dia, 'W'=semana, 'ME'=mês)
        data_col: Coluna de data/hora

    Returns:
        DataFrame clusterizado com coerência temporal
    """
    df = df.copy()
    df[data_col] = pd.to_datetime(df[data_col])
    df["_janela"] = df[data_col].dt.to_period(janela)

    resultados = []
    for periodo, grupo in df.groupby("_janela"):
        if len(grupo) == 0:
            continue
        grupo_clust = clusterizar_focos(grupo, raio_km=raio_km)
        grupo_clust["_janela"] = str(periodo)
        resultados.append(grupo_clust)

    df_final = pd.concat(resultados, ignore_index=True)
    df_final = df_final.drop(columns=["_janela"], errors="ignore")
    return df_final


def resumo_clusters(df_original: pd.DataFrame, df_clusterizado: pd.DataFrame) -> dict:
    """Retorna um dicionário com estatísticas da clusterização."""
    return {
        "registros_originais":    len(df_original),
        "registros_clusterizados": len(df_clusterizado),
        "reducao_pct": (1 - len(df_clusterizado) / len(df_original)) * 100,
    }
