# 🔥 Previsão de Incêndios no Brasil — Zetta Lab Desafio III

Projeto de Ciência de Dados para previsão e análise de severidade de focos de incêndio no Brasil,
utilizando dados do INPE/BDQueimadas, PostgreSQL + PostGIS, XGBoost e uma API Flask.

## Estrutura do Projeto

```
📁 Data Science Zetta/
├── 📁 api/
│   └── app.py              # API Flask com os endpoints de predição
├── 📁 csv/                 # ⚠️ NÃO versionado — ver instrução abaixo
│   └── dados_queimadas.csv # ~4GB — baixar do INPE
├── 📁 docs/
│   └── Plano.md            # Plano de ação do projeto
├── 📁 models/              # Modelos treinados (gerados pelo notebook)
│   ├── modelo_severidade.joblib
│   ├── modelo_ocorrencia.joblib
│   └── metadata.json
├── 📁 notebooks/
│   └── analise_incendios.ipynb  # Notebook principal (8 seções)
├── 📁 src/
│   ├── data_loader.py      # Carga de dados do PostgreSQL/CSV
│   ├── features.py         # Feature engineering (Angstrom, FMA, etc.)
│   ├── clustering.py       # DBSCAN geoespacial (raio 100km)
│   ├── climate_api.py      # Enriquecimento via NASA POWER API
│   ├── importar_dados.py   # Importação do CSV para o PostgreSQL
│   └── salvar_modelos.py   # Treina e salva os modelos para a API
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Pré-requisitos

- Python 3.11+
- PostgreSQL 18 + extensão PostGIS 3.6
- ~6GB de espaço em disco

---

## Setup

### 1. Clonar o repositório

```bash
git clone https://github.com/seu-usuario/incendios-brasil.git
cd incendios-brasil
```

### 2. Instalar dependências Python

```bash
pip install -r requirements.txt
```

### 3. Obter os dados do INPE

> ⚠️ O CSV não está no repositório (4GB). Baixe diretamente do INPE:

1. Acesse [data.inpe.br/queimadas/dados-abertos](https://data.inpe.br/queimadas/dados-abertos/)
2. Baixe os focos de incêndio dos anos desejados (2015–2025)
3. Salve como `csv/dados_queimadas.csv`

### 4. Configurar o banco de dados

```bash
# Com PostgreSQL 18 + PostGIS instalados e rodando:
psql -U postgres -c "CREATE DATABASE incendios_db;"
psql -U postgres -d incendios_db -c "CREATE EXTENSION postgis;"
```

Crie um arquivo `.env` na raiz (nunca commite este arquivo):

```env
DB_URL=postgresql://postgres:sua_senha@localhost:5432/incendios_db
```

### 5. Importar os dados para o banco

```bash
python src/importar_dados.py
```

> ⏳ Processo leva ~2h para 30M de registros

---

## Executar o Notebook

```bash
jupyter notebook notebooks/analise_incendios.ipynb
```

O notebook cobre:
| Seção | Descrição |
|-------|-----------|
| 0. Setup | Imports e configuração |
| 1. Dados | Carregamento do PostgreSQL |
| 2. Limpeza | Remoção de -999, DBSCAN 100km |
| 3. Schema | Documentação da tabela PostGIS |
| 4. Enriquecimento | NASA POWER API nos representantes de cluster |
| 5. Features | Angstrom, FMA, features temporais |
| 6. EDA | Mapas, correlações, sazonalidade |
| 7. Modelo #1 | XGBoost Regressor — Severidade FRP |
| 8. Modelo #2 | XGBoost Classifier — Ocorrência futura |

---

## Treinar e Salvar os Modelos

Após executar o notebook, ou diretamente via script:

```bash
python src/salvar_modelos.py
```

Isso gera os arquivos em `models/`:
- `modelo_severidade.joblib`
- `modelo_ocorrencia.joblib`
- `encoder_bioma.joblib`, `encoder_estado.joblib`
- `metadata.json`

---

## API Flask

### Iniciar o servidor

```bash
python api/app.py
# Servidor em: http://localhost:5000
```

### Endpoints

#### `GET /health`
Verifica status da API e dos modelos.

#### `GET /modelo/info`
Retorna metadados dos modelos (métricas, data de treino, features).

#### `POST /predict/severidade`
Prevê o FRP (Fire Radiative Power) de um foco de incêndio.

```bash
curl -X POST http://localhost:5000/predict/severidade \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": -10.5,
    "longitude": -52.0,
    "dias_sem_chuva": 45,
    "precipitacao": 0.0,
    "risco_fogo": 0.85,
    "mes": 8,
    "dia_do_ano": 220,
    "hora": 14,
    "bioma_enc": 2,
    "estado_enc": 5
  }'
```

**Resposta:**
```json
{
  "frp_previsto_mw": 38.4,
  "classe_severidade": "Medio",
  "features_recebidas": { ... }
}
```

#### `POST /predict/ocorrencia`
Prevê a probabilidade de incêndio em uma célula geoespacial 0.5° × 0.5°.

```bash
curl -X POST http://localhost:5000/predict/ocorrencia \
  -H "Content-Type: application/json" \
  -d '{
    "lat_grid": -10.0,
    "lon_grid": -52.0,
    "media_risco": 0.72,
    "media_dias_seco": 38.0,
    "media_precip": 1.2,
    "n_historico": 450,
    "mes": 8
  }'
```

**Resposta:**
```json
{
  "probabilidade_incendio": 0.87,
  "risco": "Alto",
  "cell_id": "-10.0_-52.0"
}
```

---

## Como subir ao GitHub

```bash
# Inicializar o repositório
git init
git add .
git commit -m "feat: estrutura inicial do projeto"

# Conectar ao GitHub (crie o repositório em github.com primeiro)
git remote add origin https://github.com/seu-usuario/incendios-brasil.git
git branch -M main
git push -u origin main
```

> ⚠️ O `.gitignore` já exclui automaticamente:
> - `csv/*.csv` — dados brutos (4GB, baixar do INPE)
> - `models/*.joblib` — modelos (gerados localmente após treino)
> - `.env` — credenciais do banco
> - `__pycache__/`, `.ipynb_checkpoints/` — cache

---

## Stack Tecnológica

| Camada | Ferramenta |
|--------|-----------|
| Linguagem | Python 3.11+ |
| Banco de dados | PostgreSQL 18 + PostGIS 3.6 |
| ML | XGBoost, Scikit-learn |
| API | Flask 3.0 |
| Visualização | Plotly, Folium, Seaborn |
| Dados climáticos | NASA POWER API |
| Dados de focos | INPE/BDQueimadas |

---

## Fonte dos Dados

- **Focos de incêndio:** [INPE — Dados Abertos](https://data.inpe.br/queimadas/dados-abertos/)
- **Dados climáticos:** [NASA POWER API](https://power.larc.nasa.gov/)
- **Biomas:** IBGE 2004

---

*Projeto desenvolvido para o Desafio III — Zetta Lab | Squad de Ciência de Dados*
