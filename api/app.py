"""
api/app.py
----------
Flask API que serve os modelos treinados de:
  - Previsão de Severidade (FRP) — Regressão
  - Previsão de Ocorrência    — Classificação Binária

Endpoints:
  GET  /health                → status da API
  POST /predict/severidade    → prevê o FRP (MW) de um foco
  POST /predict/ocorrencia    → prevê probabilidade de incêndio em uma célula
  GET  /modelo/info           → metadados dos modelos carregados
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from datetime import datetime

# Adicionar src/ ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from features import indice_angstrom, add_grid_cell

app = Flask(__name__)

# ── Carregar modelos ao iniciar ──────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

try:
    modelo_sev  = joblib.load(os.path.join(MODELS_DIR, 'modelo_severidade.joblib'))
    modelo_prev = joblib.load(os.path.join(MODELS_DIR, 'modelo_ocorrencia.joblib'))
    meta        = json.load(open(os.path.join(MODELS_DIR, 'metadata.json')))
    print("[OK] Modelos carregados com sucesso")
except FileNotFoundError as e:
    print(f"[AVISO] Modelos nao encontrados: {e}")
    print("        Execute o notebook para treinar e salvar os modelos.")
    modelo_sev  = None
    modelo_prev = None
    meta        = {}

# Features esperadas por cada modelo
FEATS_SEV = [
    'latitude', 'longitude', 'dias_sem_chuva', 'precipitacao',
    'risco_fogo', 'mes', 'dia_do_ano', 'hora', 'bioma_enc', 'estado_enc'
]
FEATS_CLF = [
    'lat_grid', 'lon_grid', 'media_risco', 'media_dias_seco',
    'media_precip', 'n_historico', 'mes'
]


# ── Helpers ──────────────────────────────────────────────────────────────────
def _erro(msg: str, code: int = 400):
    return jsonify({'erro': msg}), code

def _checar_modelo(modelo, nome: str):
    if modelo is None:
        return _erro(f'Modelo {nome} nao esta carregado. Treine e salve via notebook.', 503)
    return None


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get('/health')
def health():
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.utcnow().isoformat(),
        'modelos': {
            'severidade': modelo_sev is not None,
            'ocorrencia': modelo_prev is not None,
        }
    })


@app.get('/modelo/info')
def modelo_info():
    return jsonify(meta if meta else {'aviso': 'metadata.json nao encontrado'})


@app.post('/predict/severidade')
def predict_severidade():
    """
    Prevê o FRP (Fire Radiative Power, em MW) de um foco de incêndio.

    Payload JSON esperado:
    {
        "latitude":       -10.5,
        "longitude":      -52.0,
        "dias_sem_chuva": 45,
        "precipitacao":   0.0,
        "risco_fogo":     0.85,
        "mes":            8,
        "dia_do_ano":     220,
        "hora":           14,
        "bioma_enc":      2,
        "estado_enc":     5
    }

    Resposta:
    {
        "frp_previsto_mw": 38.4,
        "classe_severidade": "Médio",
        "features_recebidas": {...}
    }
    """
    err = _checar_modelo(modelo_sev, 'severidade')
    if err:
        return err

    data = request.get_json(force=True)
    if not data:
        return _erro('Payload JSON ausente')

    # Verificar campos obrigatórios
    faltando = [f for f in FEATS_SEV if f not in data]
    if faltando:
        return _erro(f'Campos ausentes: {faltando}')

    try:
        X = pd.DataFrame([{f: float(data[f]) for f in FEATS_SEV}])
        X = X.fillna(-1)

        frp = float(modelo_sev.predict(X)[0])

        # Classificar severidade
        if frp < 10:
            classe = 'Baixo'
        elif frp < 50:
            classe = 'Medio'
        elif frp < 200:
            classe = 'Alto'
        else:
            classe = 'Extremo'

        return jsonify({
            'frp_previsto_mw':   round(frp, 2),
            'classe_severidade': classe,
            'features_recebidas': {f: data[f] for f in FEATS_SEV}
        })

    except Exception as e:
        return _erro(f'Erro na predicao: {str(e)}', 500)


@app.post('/predict/ocorrencia')
def predict_ocorrencia():
    """
    Prevê a probabilidade de ocorrência de incêndio em uma célula geoespacial.

    Payload JSON esperado:
    {
        "lat_grid":         -10.0,
        "lon_grid":         -52.0,
        "media_risco":      0.72,
        "media_dias_seco":  38.0,
        "media_precip":     1.2,
        "n_historico":      450,
        "mes":              8
    }

    Resposta:
    {
        "probabilidade_incendio": 0.87,
        "risco": "Alto",
        "cell_id": "-10.0_-52.0"
    }
    """
    err = _checar_modelo(modelo_prev, 'ocorrencia')
    if err:
        return err

    data = request.get_json(force=True)
    if not data:
        return _erro('Payload JSON ausente')

    faltando = [f for f in FEATS_CLF if f not in data]
    if faltando:
        return _erro(f'Campos ausentes: {faltando}')

    try:
        X = pd.DataFrame([{f: float(data[f]) for f in FEATS_CLF}])
        X = X.fillna(-1)

        prob = float(modelo_prev.predict_proba(X)[0][1])

        # Classificar risco
        if prob < 0.3:
            risco = 'Baixo'
        elif prob < 0.6:
            risco = 'Moderado'
        elif prob < 0.8:
            risco = 'Alto'
        else:
            risco = 'Critico'

        return jsonify({
            'probabilidade_incendio': round(prob, 4),
            'risco':   risco,
            'cell_id': f"{data['lat_grid']}_{data['lon_grid']}"
        })

    except Exception as e:
        return _erro(f'Erro na predicao: {str(e)}', 500)


# ── Iniciar ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
