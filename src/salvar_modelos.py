"""
src/salvar_modelos.py
----------------------
Salva os modelos treinados e metadados na pasta models/.
Execute este script após treinar os modelos no notebook.

Uso:
    python src/salvar_modelos.py
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from xgboost import XGBRegressor, XGBClassifier

import sys
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_sample_from_db, save_parquet
from features import add_temporal_features, add_grid_cell, classificar_frp

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

FEATS_SEV = [
    'latitude', 'longitude', 'dias_sem_chuva', 'precipitacao',
    'risco_fogo', 'mes', 'dia_do_ano', 'hora', 'bioma_enc', 'estado_enc'
]
FEATS_CLF = [
    'lat_grid', 'lon_grid', 'media_risco', 'media_dias_seco',
    'media_precip', 'n_historico', 'mes'
]


def treinar_e_salvar():
    print("Carregando amostra do banco...")
    df = load_sample_from_db(n=500_000)
    df['data_hora'] = pd.to_datetime(df['data_hora'])
    df = df.replace(-999, np.nan).replace(-999.0, np.nan)
    df = df.dropna(subset=['latitude', 'longitude']).drop_duplicates(subset=['id'])
    df = add_temporal_features(df)
    df = add_grid_cell(df, resolucao=0.5)

    le_bioma  = LabelEncoder()
    le_estado = LabelEncoder()
    df['bioma_enc']  = le_bioma.fit_transform(df['bioma'].fillna('Desconhecido'))
    df['estado_enc'] = le_estado.fit_transform(df['estado'].fillna('Desconhecido'))

    # ── Modelo 1: Severidade (FRP) ──────────────────────────────────────────
    print("\nTreinando Modelo #1 — Severidade (FRP)...")
    df_m = df.dropna(subset=['frp']).copy()
    X = df_m[FEATS_SEV].fillna(-1)
    y = df_m['frp']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo_sev = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    modelo_sev.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    y_pred = modelo_sev.predict(X_te)
    mae_sev = mean_absolute_error(y_te, y_pred)
    r2_sev  = r2_score(y_te, y_pred)
    print(f"  MAE: {mae_sev:.2f} MW | R2: {r2_sev:.4f}")

    joblib.dump(modelo_sev, os.path.join(MODELS_DIR, 'modelo_severidade.joblib'))
    print("  Salvo: models/modelo_severidade.joblib")

    # ── Modelo 2: Ocorrência (Classificação) ───────────────────────────────
    print("\nTreinando Modelo #2 — Ocorrencia...")
    df_grid = add_grid_cell(df.copy())
    df_grid['semana'] = df_grid['data_hora'].dt.to_period('W')

    focos_cel = (
        df_grid.groupby(['cell_id', 'semana']).size()
        .reset_index(name='n_focos')
    )
    focos_cel['incendio'] = (focos_cel['n_focos'] > 0).astype(int)

    stats_cel = df_grid.groupby('cell_id').agg(
        lat_grid=('lat_grid', 'first'), lon_grid=('lon_grid', 'first'),
        media_risco=('risco_fogo', 'mean'),
        media_dias_seco=('dias_sem_chuva', 'mean'),
        media_precip=('precipitacao', 'mean'),
        n_historico=('id', 'count')
    ).reset_index()

    df_clf = focos_cel.merge(stats_cel, on='cell_id')
    df_clf['mes'] = df_clf['semana'].dt.start_time.dt.month

    pos = df_clf['incendio'].sum()
    neg = (df_clf['incendio'] == 0).sum()
    X_c = df_clf[FEATS_CLF].fillna(-1)
    y_c = df_clf['incendio']
    X_tc, X_vc, y_tc, y_vc = train_test_split(
        X_c, y_c, test_size=0.2, random_state=42, stratify=y_c
    )

    modelo_prev = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=neg / pos if pos > 0 else 1,
        eval_metric='logloss', random_state=42, n_jobs=-1
    )
    modelo_prev.fit(X_tc, y_tc, eval_set=[(X_vc, y_vc)], verbose=False)

    y_prob = modelo_prev.predict_proba(X_vc)[:, 1]
    auc = roc_auc_score(y_vc, y_prob)
    print(f"  AUC-ROC: {auc:.4f}")

    joblib.dump(modelo_prev, os.path.join(MODELS_DIR, 'modelo_ocorrencia.joblib'))
    print("  Salvo: models/modelo_ocorrencia.joblib")

    # ── Salvar encoders ─────────────────────────────────────────────────────
    joblib.dump(le_bioma,  os.path.join(MODELS_DIR, 'encoder_bioma.joblib'))
    joblib.dump(le_estado, os.path.join(MODELS_DIR, 'encoder_estado.joblib'))

    # ── Metadados ───────────────────────────────────────────────────────────
    metadata = {
        'treinado_em': datetime.utcnow().isoformat(),
        'registros_treino': len(df),
        'modelo_severidade': {
            'algoritmo': 'XGBRegressor',
            'n_estimators': 300,
            'features': FEATS_SEV,
            'mae_mw': round(mae_sev, 2),
            'r2':     round(r2_sev, 4),
        },
        'modelo_ocorrencia': {
            'algoritmo': 'XGBClassifier',
            'n_estimators': 300,
            'features': FEATS_CLF,
            'auc_roc': round(auc, 4),
        },
    }
    with open(os.path.join(MODELS_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print("\nSalvo: models/metadata.json")
    print("\n[OK] Todos os modelos e encoders salvos em models/")
    return metadata


if __name__ == '__main__':
    treinar_e_salvar()
