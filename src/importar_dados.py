"""
importar_dados.py
-----------------
Importa o CSV de focos de incêndio (4GB) para o PostgreSQL em chunks,
aplicando limpeza e feature engineering durante a carga.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from tqdm import tqdm
import os
import time

# ── Configurações ────────────────────────────────────────────────────────────
CSV_PATH    = "csv/dados_queimadas.csv"
DB_URL      = "postgresql://postgres:postgres@localhost:5432/incendios_db"
TABLE_NAME  = "focos_incendio"
CHUNK_SIZE  = 500_000
ENCODING    = "utf-8"

# ── Mapeamento de estações (hemisfério sul — Brasil) ─────────────────────────
ESTACAO_MAP = {
    12: "Verão",  1: "Verão",   2: "Verão",
     3: "Outono", 4: "Outono",  5: "Outono",
     6: "Inverno",7: "Inverno", 8: "Inverno",
     9: "Primavera",10:"Primavera",11:"Primavera"
}

# ── Funções de Feature Engineering ──────────────────────────────────────────
def indice_angstrom(umidade, temperatura):
    """Índice de Angstrom — risco de fogo."""
    return (umidade / 20.0) + ((27.0 - temperatura) / 10.0)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features temporais e geoespaciais ao chunk."""
    df = df.copy()

    # Converter data_hora
    df["data_hora"] = pd.to_datetime(df["data_hora"], errors="coerce")

    # Features temporais
    df["mes"]       = df["data_hora"].dt.month
    df["estacao"]   = df["mes"].map(ESTACAO_MAP)
    df["dia_do_ano"]= df["data_hora"].dt.dayofyear
    df["hora"]      = df["data_hora"].dt.hour

    # Inicializar colunas que serão enriquecidas depois
    df["umidade"]    = np.nan
    df["temperatura"]= np.nan
    df["fma"]        = np.nan
    df["angstrom"]   = np.nan   # sem umidade/temp ainda

    # Coluna geom como WKT para o PostGIS
    df["geom"] = df.apply(
        lambda r: f"SRID=4326;POINT({r['longitude']} {r['latitude']})"
        if pd.notna(r["latitude"]) and pd.notna(r["longitude"]) else None,
        axis=1
    )

    return df

def limpar(df: pd.DataFrame) -> pd.DataFrame:
    """Limpeza básica: remove -999, duplicatas, sem lat/lon."""
    df = df.replace(-999, np.nan).replace(-999.0, np.nan)
    df = df.dropna(subset=["latitude", "longitude"])
    df = df.drop_duplicates(subset=["id"])
    return df

def renomear_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza nomes de colunas para o schema do banco."""
    return df.rename(columns={"dia_sem_chuva": "dias_sem_chuva"})

# ── Importação principal ─────────────────────────────────────────────────────
def importar():
    engine = create_engine(DB_URL, pool_pre_ping=True)

    # Contar total de linhas para a barra de progresso
    print("Contando linhas do CSV (pode demorar ~30s para 4GB)...")
    total_lines = sum(1 for _ in open(CSV_PATH, encoding=ENCODING)) - 1
    total_chunks = (total_lines // CHUNK_SIZE) + 1
    print(f"Total de registros: {total_lines:,}  |  Chunks: {total_chunks}")

    start = time.time()
    total_importado = 0
    total_ignorado  = 0

    reader = pd.read_csv(
        CSV_PATH,
        chunksize=CHUNK_SIZE,
        encoding=ENCODING,
        low_memory=False,
    )

    with tqdm(total=total_chunks, desc="Importando", unit="chunk") as pbar:
        for i, chunk in enumerate(reader):
            try:
                chunk = renomear_colunas(chunk)
                chunk = limpar(chunk)
                chunk = add_features(chunk)

                # Selecionar apenas colunas do schema
                cols = [
                    "id", "data_hora", "satelite", "pais", "estado",
                    "municipio", "bioma", "dias_sem_chuva", "precipitacao",
                    "risco_fogo", "frp", "latitude", "longitude",
                    "umidade", "temperatura", "angstrom", "fma",
                    "mes", "estacao", "dia_do_ano", "hora"
                ]
                chunk = chunk[[c for c in cols if c in chunk.columns]]

                chunk.to_sql(
                    TABLE_NAME,
                    engine,
                    if_exists="append",
                    index=False,
                    method="multi",
                    chunksize=10_000,
                )

                total_importado += len(chunk)

            except Exception as e:
                print(f"\n⚠️  Erro no chunk {i}: {e}")
                total_ignorado += CHUNK_SIZE

            pbar.set_postfix({
                "importados": f"{total_importado:,}",
                "ignorados":  f"{total_ignorado:,}"
            })
            pbar.update(1)

    elapsed = time.time() - start

    # Atualizar coluna geom via SQL depois da carga (mais rápido)
    print("\nAtualizando coluna geom (ST_MakePoint)...")
    with engine.connect() as conn:
        conn.execute(text("""
            UPDATE focos_incendio
            SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)
            WHERE geom IS NULL AND latitude IS NOT NULL AND longitude IS NOT NULL;
        """))
        conn.commit()

    print(f"\n✅ Importação concluída em {elapsed/60:.1f} min")
    print(f"   Registros importados : {total_importado:,}")
    print(f"   Registros ignorados  : {total_ignorado:,}")

    # Verificar contagem final
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM focos_incendio")).scalar()
        print(f"   Total no banco       : {count:,}")

if __name__ == "__main__":
    importar()
