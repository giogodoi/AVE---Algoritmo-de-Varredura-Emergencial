"""
src/data_loader.py
------------------
Funções para carregar e salvar dados do projeto de incêndios.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from pathlib import Path

DB_URL   = "postgresql://postgres:postgres@localhost:5432/incendios_db"
CSV_PATH = "csv/dados_queimadas.csv"


def get_engine():
    """Retorna uma engine SQLAlchemy conectada ao banco incendios_db."""
    return create_engine(DB_URL, pool_pre_ping=True)


def load_sample_from_db(n: int = 500_000, random_state: int = 42) -> pd.DataFrame:
    """
    Carrega uma amostra aleatória de n registros do PostgreSQL.
    Usa TABLESAMPLE para eficiência em tabelas grandes.
    """
    engine = get_engine()
    fraction = min(100.0, (n / 30_000_000) * 100 * 1.5)  # oversample um pouco
    query = f"""
        SELECT * FROM focos_incendio
        TABLESAMPLE SYSTEM({fraction:.4f})
        LIMIT {n};
    """
    df = pd.read_sql(query, engine)
    df["data_hora"] = pd.to_datetime(df["data_hora"])
    return df


def load_full_from_db(query: str = "SELECT * FROM focos_incendio") -> pd.DataFrame:
    """Carrega dados completos do banco — use com cautela (30M+ registros)."""
    engine = get_engine()
    return pd.read_sql(query, engine)


def load_chunks_from_csv(chunksize: int = 500_000):
    """Gerador de chunks do CSV original."""
    return pd.read_csv(
        CSV_PATH,
        chunksize=chunksize,
        encoding="utf-8",
        low_memory=False,
    )


def save_parquet(df: pd.DataFrame, path: str):
    """Salva DataFrame como Parquet comprimido."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="snappy")
    print(f"✅ Salvo: {path}  ({Path(path).stat().st_size / 1e6:.1f} MB)")


def load_parquet(path: str) -> pd.DataFrame:
    """Carrega um arquivo Parquet."""
    df = pd.read_parquet(path)
    if "data_hora" in df.columns:
        df["data_hora"] = pd.to_datetime(df["data_hora"])
    return df


def get_db_stats() -> dict:
    """Retorna estatísticas básicas do banco."""
    engine = get_engine()
    with engine.connect() as conn:
        total    = conn.execute(text("SELECT COUNT(*) FROM focos_incendio")).scalar()
        biomas   = conn.execute(text("SELECT COUNT(DISTINCT bioma) FROM focos_incendio")).scalar()
        estados  = conn.execute(text("SELECT COUNT(DISTINCT estado) FROM focos_incendio")).scalar()
        min_data = conn.execute(text("SELECT MIN(data_hora) FROM focos_incendio")).scalar()
        max_data = conn.execute(text("SELECT MAX(data_hora) FROM focos_incendio")).scalar()
    return {
        "total_registros": total,
        "biomas": biomas,
        "estados": estados,
        "periodo_inicio": min_data,
        "periodo_fim": max_data,
    }
