"""
src/climate_api.py
------------------
Enriquecimento com dados climáticos via NASA POWER API (fonte principal).
Estratégia: clusterizar primeiro → enriquecer apenas os representantes.

Fluxo recomendado:
    1. df_clust = clusterizar_focos(df, raio_km=100)   # reduz volume drasticamente
    2. df_enriq = enriquecer_representantes(df_clust)  # chama API só nos centróides
    3. df_final = df.merge(df_enriq[...], on='cluster_id')  # propaga p/ todos
"""

import requests
import time
import pandas as pd
import numpy as np
from functools import lru_cache


# ── NASA POWER API (fonte principal — gratuita, sem chave) ──────────────────
BASE_URL_NASA = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
    "?parameters=T2M,RH2M"
    "&community=RE"
    "&longitude={lon}&latitude={lat}"
    "&start={data}&end={data}"
    "&format=JSON"
)


@lru_cache(maxsize=100_000)
def buscar_nasa_power(lat_r: float, lon_r: float, data_str: str) -> tuple:
    """
    Busca temperatura (T2M °C) e umidade relativa (RH2M %)
    via NASA POWER API para coordenada arredondada + data.

    Coordenadas devem ser arredondadas a 0.5° (resolução da NASA POWER)
    antes de chamar esta função — isso maximiza o cache hit rate.

    Args:
        lat_r    : Latitude arredondada a 0.5°  (ex: -10.0, -10.5)
        lon_r    : Longitude arredondada a 0.5° (ex: -52.0, -52.5)
        data_str : Data no formato 'YYYYMMDD'   (ex: '20230815')

    Returns:
        (temperatura_C, umidade_pct) ou (nan, nan) em caso de erro
    """
    url = BASE_URL_NASA.format(lat=lat_r, lon=lon_r, data=data_str)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        props = resp.json()["properties"]["parameter"]
        temp = list(props["T2M"].values())[0]
        umid = list(props["RH2M"].values())[0]
        # NASA usa -999 para dado inválido
        temp = np.nan if temp == -999 else float(temp)
        umid = np.nan if umid == -999 else float(umid)
        return temp, umid
    except Exception:
        return np.nan, np.nan


def _arredondar_coord(val: float, resolucao: float = 0.5) -> float:
    """Arredonda coordenada para a grade da NASA POWER (padrão 0.5°)."""
    return round(val / resolucao) * resolucao


def enriquecer_representantes(
    df_clust: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    data_col: str = "data_hora",
    delay_s: float = 0.3,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Enriquece os pontos representantes de clusters com temperatura e umidade
    via NASA POWER API.

    Deve ser chamada APÓS a clusterização — recebe apenas os representantes
    (1 por cluster), não o dataset completo. Isso reduz chamadas à API
    de milhões para alguns milhares.

    Exemplo de uso:
        df_clust = clusterizar_focos(df, raio_km=100)
        df_clust = enriquecer_representantes(df_clust)
        # depois propagar para o df completo via cluster_id

    Args:
        df_clust : DataFrame com representantes dos clusters
        lat_col  : Coluna de latitude
        lon_col  : Coluna de longitude
        data_col : Coluna de data/hora
        delay_s  : Intervalo entre chamadas para não sobrecarregar a API
        verbose  : Mostrar progresso

    Returns:
        df_clust com colunas 'temperatura' e 'umidade' preenchidas
    """
    df = df_clust.copy()
    df[data_col] = pd.to_datetime(df[data_col])

    # Coordenadas arredondadas para maximizar cache hit rate
    df["_lat_r"] = df[lat_col].apply(_arredondar_coord)
    df["_lon_r"] = df[lon_col].apply(_arredondar_coord)
    df["_data_str"] = df[data_col].dt.strftime("%Y%m%d")

    # Combinações únicas (lat, lon, data) — mesmo com lru_cache, evita
    # chamadas desnecessárias quando há duplicatas após o arredondamento
    chaves_unicas = df[["_lat_r", "_lon_r", "_data_str"]].drop_duplicates()
    total = len(chaves_unicas)

    if verbose:
        print(f"Chamadas únicas à NASA POWER API: {total:,}")
        print(f"  (do total de {len(df):,} representantes de cluster)")

    # Preencher cache com as chaves únicas
    cache_resultados = {}
    for i, (_, row) in enumerate(chaves_unicas.iterrows()):
        chave = (row["_lat_r"], row["_lon_r"], row["_data_str"])
        cache_resultados[chave] = buscar_nasa_power(*chave)
        if verbose and (i + 1) % 100 == 0:
            pct = (i + 1) / total * 100
            print(f"  {i+1:,}/{total:,} ({pct:.0f}%)")
        time.sleep(delay_s)

    # Aplicar resultados ao DataFrame
    df["temperatura"] = df.apply(
        lambda r: cache_resultados.get(
            (r["_lat_r"], r["_lon_r"], r["_data_str"]), (np.nan, np.nan)
        )[0],
        axis=1,
    )
    df["umidade"] = df.apply(
        lambda r: cache_resultados.get(
            (r["_lat_r"], r["_lon_r"], r["_data_str"]), (np.nan, np.nan)
        )[1],
        axis=1,
    )

    df = df.drop(columns=["_lat_r", "_lon_r", "_data_str"])
    if verbose:
        print(f"Temperatura media: {df['temperatura'].mean():.1f} C")
        print(f"Umidade media    : {df['umidade'].mean():.1f} %")
        print(f"Cobertura temp.  : {df['temperatura'].notna().mean()*100:.1f}%")

    return df


def propagar_clima_para_df(
    df_completo: pd.DataFrame,
    df_clust_enriq: pd.DataFrame,
    cluster_col: str = "cluster_id",
) -> pd.DataFrame:
    """
    Propaga temperatura e umidade dos representantes de cluster
    para todos os registros do DataFrame completo.

    Args:
        df_completo    : DataFrame completo (todos os focos)
        df_clust_enriq : Representantes enriquecidos com temperatura/umidade
        cluster_col    : Coluna de ID do cluster

    Returns:
        df_completo com colunas 'temperatura' e 'umidade' preenchidas
    """
    mapa = df_clust_enriq[[cluster_col, "temperatura", "umidade"]].drop_duplicates(cluster_col)
    return df_completo.merge(mapa, on=cluster_col, how="left")
