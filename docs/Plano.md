# 🔥 Plano de Ação — Ciência de Dados: Previsão e Severidade de Incêndios

> **Projeto:** Zetta Lab — Desafio III | Squad de Ciência de Dados
> **Base de Dados:** BDQueimadas / BDMAPAS (INPE) — últimos 10 anos
> **Ambiente:** Python + Jupyter Notebook

***

## Fase 1 — Aquisição dos Dados

### 1.1 Fonte e Download

- Acessar o portal [BDQueimadas (INPE)](https://terrabrasilis.dpi.inpe.br/queimadas/bdqueimadas/)  e/ou [data.inpe.br](https://data.inpe.br)[^12][^13]
- Baixar os CSVs anuais dos últimos 10 anos (2015–2025) de focos de incêndio no Brasil[^1]
- Os arquivos contêm os campos confirmados pelo INPE:[^10]

| Campo | Tipo | Descrição |
| :-- | :-- | :-- |
| `DataHora` | string | Data/hora da detecção (GMT) |
| `Satelite` | string | Satélite/algoritmo detector |
| `Pais` | string | País |
| `Estado` | string | Estado |
| `Municipio` | string | Município |
| `Bioma` | string | Bioma (IBGE 2004) |
| `DiaSemChuva` | integer | Dias acumulados sem chuva |
| `Precipitacao` | double | Precipitação acumulada no dia |
| `RiscoFogo` | double | Risco de fogo (0.0 a 1.0) |
| `Latitude` | double | Latitude decimal |
| `Longitude` | double | Longitude decimal |
| `FRP` | double | Fire Radiative Power (MW) — severidade |

### 1.2 Volume Estimado

- ~30 milhões de registros totais ao longo de 10 anos
- Processar em chunks com `pandas` (`chunksize=500_000`) para evitar estouro de memória

***

## Fase 2 — Importação para Banco de Dados

### 2.1 Escolha do Banco

Recomenda-se **PostgreSQL + PostGIS** (extensão geoespacial), que permite consultas por raio geográfico de forma nativa e eficiente.

```bash
# Instalação da extensão PostGIS
CREATE EXTENSION postgis;
```


### 2.2 Schema da Tabela Principal

```sql
CREATE TABLE focos_incendio (
    id              SERIAL PRIMARY KEY,
    data_hora       TIMESTAMP,
    satelite        VARCHAR(20),
    pais            VARCHAR(30),
    estado          VARCHAR(30),
    municipio       VARCHAR(60),
    bioma           VARCHAR(30),
    dias_sem_chuva  INTEGER,
    precipitacao    DOUBLE PRECISION,
    risco_fogo      DOUBLE PRECISION,
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    area_industrial BOOLEAN,
    frp             DOUBLE PRECISION,
    umidade         DOUBLE PRECISION,   -- a ser enriquecido
    temperatura     DOUBLE PRECISION,   -- a ser enriquecido
    geom            GEOMETRY(Point, 4326)
);

-- Índice geoespacial para consultas rápidas por raio
CREATE INDEX idx_focos_geom ON focos_incendio USING GIST(geom);
CREATE INDEX idx_focos_data ON focos_incendio (data_hora);
```


### 2.3 Importação via Python

```python
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql://user:senha@localhost:5432/incendios_db")

for chunk in pd.read_csv("focos_2015_2025.csv", chunksize=500_000):
    chunk.to_sql("focos_incendio", engine, if_exists="append", index=False)
```


***

## Fase 3 — Limpeza de Dados

### 3.1 Tratamento de Valores Inválidos

O INPE usa `-999` como valor inválido nos campos numéricos. Substituir por `NaN` e tratar:[^10]

```python
import pandas as pd
import numpy as np

df.replace(-999, np.nan, inplace=True)

# Remover registros sem lat/lon
df.dropna(subset=["Latitude", "Longitude"], inplace=True)

# Remover duplicatas exatas
df.drop_duplicates(inplace=True)
```


### 3.2 Clusterização por Raio de 100km (Redução de Registros)

A função a seguir agrupa focos dentro de um raio de **100 km** em um único registro representativo, reduzindo drasticamente o volume de dados:

```python
from sklearn.cluster import DBSCAN
import numpy as np

def clusterizar_focos(df, raio_km=100):
    """
    Agrupa focos de incêndio dentro de um raio de até 100km
    usando DBSCAN com métrica haversine.
    Retorna um DataFrame com um ponto representativo por cluster.
    """
    coords = np.radians(df[["Latitude", "Longitude"]].values)
    
    # Epsilon em radianos: 100km / 6371km (raio da Terra)
    epsilon = raio_km / 6371.0
    
    db = DBSCAN(
        eps=epsilon,
        min_samples=1,
        algorithm="ball_tree",
        metric="haversine"
    ).fit(coords)
    
    df["cluster_id"] = db.labels_
    
    # Agregar: usar o ponto de maior FRP como representante do cluster
    df_agrupado = (
        df.sort_values("FRP", ascending=False)
          .groupby("cluster_id")
          .first()
          .reset_index(drop=True)
    )
    
    return df_agrupado

df_limpo = clusterizar_focos(df)
print(f"Redução: {len(df)} → {len(df_limpo)} registros")
```

> **Nota:** Processar por janela temporal (ex: por dia ou por semana) antes de clusterizar para garantir coerência temporal dos agrupamentos.

***

## Fase 4 — Enriquecimento com Dados Climáticos

### 4.1 APIs Recomendadas

| API | Variáveis | Acesso |
| :-- | :-- | :-- |
| **NASA POWER** | Temperatura, umidade, precipitação, radiação | Gratuito, REST |
| **Copernicus ERA5** (ECMWF) | Temperatura 2m, umidade relativa, vento | Gratuito, CDS API |
| **Open-Meteo** | Temperatura, umidade (histórico) | Gratuito, sem chave |

### 4.2 Implementação com NASA POWER API

```python
import requests
import time

def buscar_dados_climaticos_nasa(lat, lon, data):
    """
    Busca temperatura e umidade via NASA POWER API
    para uma coordenada e data específicas.
    """
    data_fmt = data.strftime("%Y%m%d")
    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=T2M,RH2M"
        f"&community=RE"
        f"&longitude={lon}&latitude={lat}"
        f"&start={data_fmt}&end={data_fmt}"
        f"&format=JSON"
    )
    
    resp = requests.get(url, timeout=30)
    data_json = resp.json()
    
    props = data_json["properties"]["parameter"]
    temperatura = list(props["T2M"].values())[^0]
    umidade     = list(props["RH2M"].values())[^0]
    
    return temperatura, umidade

# Aplicar ao DataFrame (usar cache para coordenadas repetidas)
# Recomendado: paralelizar com concurrent.futures ou joblib
```


### 4.3 Estratégia de Cache

Para evitar chamadas repetidas à API (coordenadas próximas e mesma data):

```python
from functools import lru_cache

@lru_cache(maxsize=100_000)
def buscar_climatico_cached(lat_round, lon_round, data_str):
    return buscar_dados_climaticos_nasa(lat_round, lon_round, data_str)

# Arredondar coordenadas a 0.5° (resolução da NASA POWER)
df["lat_r"] = df["Latitude"].round(1)
df["lon_r"] = df["Longitude"].round(1)
```


***

## Fase 5 — Feature Engineering

### 5.1 Índices Meteorológicos para Risco de Fogo

#### Índice de Angstrom (IA)

$$
IA = \frac{H}{20} + \frac{(27 - T)}{10}
$$

onde **H** = umidade relativa (%) e **T** = temperatura (°C).

- IA < 2.0 → risco muito alto
- 2.0–3.9 → risco alto
- 4.0–5.9 → risco moderado
- ≥ 6.0 → sem risco

```python
def indice_angstrom(umidade, temperatura):
    return (umidade / 20) + ((27 - temperatura) / 10)

df["angstrom"] = df.apply(
    lambda r: indice_angstrom(r["umidade"], r["temperatura"]), axis=1
)
```


#### Fórmula de Monte Alegre (FMA)

$$
FMA = \sum_{i=1}^{n} \frac{H_i}{(13 - H_{i-1})}
$$

A FMA é acumulativa — reiniciada quando há precipitação ≥ 13mm. Implementar como função iterativa por série temporal por localidade.

```python
def calcular_fma(series_umidade, series_precip, limiar_mm=13):
    fma_vals = []
    acum = 0
    for h, p in zip(series_umidade, series_precip):
        if p >= limiar_mm:
            acum = 0
        else:
            acum += h / (13 - h + 1e-6)
        fma_vals.append(acum)
    return fma_vals
```


### 5.2 Features Temporais

```python
df["mes"]          = df["data_hora"].dt.month
df["estacao"]      = df["mes"].map({
    12: "Verão", 1: "Verão", 2: "Verão",
    3: "Outono", 4: "Outono", 5: "Outono",
    6: "Inverno", 7: "Inverno", 8: "Inverno",
    9: "Primavera", 10: "Primavera", 11: "Primavera"
})
df["dia_do_ano"]   = df["data_hora"].dt.dayofyear
df["hora"]         = df["data_hora"].dt.hour
```


***

## Fase 6 — Modelo \#1: Previsão de Severidade

**Objetivo:** Prever o **FRP** (Fire Radiative Power) — índice de severidade do incêndio.

### 6.1 Features de Entrada

- Latitude, Longitude
- Umidade relativa
- Temperatura
- Dias sem chuva
- Precipitação
- Mês / Estação do ano
- Índice de Angstrom
- FMA acumulado
- Bioma (one-hot encoding)


### 6.2 Target

- `FRP` (contínuo) → **Regressão**
- Ou `classe_severidade` (baixo/médio/alto/extremo) → **Classificação**


### 6.3 Algoritmos Recomendados

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

X = df[[
    "latitude", "longitude", "umidade", "temperatura",
    "dias_sem_chuva", "precipitacao", "mes", "angstrom", "fma",
    # biomas dummies...
]]
y = df["frp"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo_severidade = XGBRegressor(n_estimators=300, max_depth=6, random_state=42)
modelo_severidade.fit(X_train, y_train)

y_pred = modelo_severidade.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R²:  {r2_score(y_test, y_pred):.4f}")
```


### 6.4 Visualizações no Notebook

- Mapa de calor da severidade prevista por região (Plotly + Mapbox)
- Feature importance (SHAP values ou barplot)
- Scatter plot: FRP real vs. FRP previsto
- Curva de aprendizado do modelo

***

## Fase 7 — Modelo \#2: Previsão de Incêndios Futuros

**Objetivo:** Classificar se uma determinada região/data **terá ou não ocorrência de incêndio**.

### 7.1 Abordagem

Transformar o problema em **classificação binária** por célula de grade geoespacial (ex: grid de 0.5° × 0.5°):

- `1` = ocorreu pelo menos um foco no período
- `0` = não ocorreu foco


### 7.2 Features de Entrada

- Histórico de focos nos últimos 30/60/90 dias na célula
- Médias históricas de temperatura, umidade, precipitação
- Dias sem chuva acumulados
- Estação do ano
- Bioma dominante na célula
- FMA médio da célula


### 7.3 Algoritmos Recomendados

```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

modelo_previsao = XGBClassifier(
    n_estimators=500,
    scale_pos_weight=neg/pos,  # balancear classes
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
modelo_previsao.fit(X_train, y_train)

y_prob = modelo_previsao.predict_proba(X_test)[:, 1]
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")
print(classification_report(y_test, modelo_previsao.predict(X_test)))
```


### 7.4 Séries Temporais (Opcional Avançado)

Para regiões com histórico denso, usar **LSTM (Keras)** ou **Prophet (Meta)** para capturar sazonalidade:

```python
from prophet import Prophet

df_prophet = df_regiao[["data_hora", "frp"]].rename(
    columns={"data_hora": "ds", "frp": "y"}
)
model = Prophet(yearly_seasonality=True, weekly_seasonality=False)
model.fit(df_prophet)
futuro = model.make_future_dataframe(periods=90)
forecast = model.predict(futuro)
model.plot(forecast)
```


### 7.5 Visualizações no Notebook

- Mapa de probabilidade de incêndio futuro por município/bioma
- Heatmap temporal (mês × estado × probabilidade)
- Matriz de confusão
- Curva ROC
- Animação temporal dos focos históricos (Plotly Express)

***

## Fase 8 — Estrutura do Notebook Jupyter

```
📓 analise_incendios.ipynb
│
├── 0. Setup & Imports
├── 1. Carregamento dos Dados (CSV → DataFrame)
├── 2. Limpeza e Pré-processamento
│   ├── 2.1 Tratamento de valores inválidos (-999)
│   └── 2.2 Clusterização por raio de 100km (DBSCAN)
├── 3. Importação para PostgreSQL
├── 4. Enriquecimento (NASA POWER API)
├── 5. Feature Engineering
│   ├── 5.1 Índice de Angstrom
│   ├── 5.2 Fórmula de Monte Alegre (FMA)
│   └── 5.3 Features temporais
├── 6. Análise Exploratória (EDA)
│   ├── Distribuição por bioma, estado, mês
│   ├── Correlação entre variáveis
│   └── Mapas de focos históricos
├── 7. Modelo #1 — Severidade (FRP)
│   ├── Treino / Teste / Validação
│   ├── Métricas (MAE, R²)
│   └── Visualizações
└── 8. Modelo #2 — Previsão Futura
    ├── Criação do grid geoespacial
    ├── Treino / Teste / Validação
    ├── Métricas (AUC-ROC, F1)
    └── Mapa de probabilidade futura
```


***

## Fase 9 — Stack Tecnológica

| Camada | Ferramenta |
| :-- | :-- |
| Linguagem | Python 3.11+ |
| Manipulação de dados | Pandas, NumPy |
| Banco de dados | PostgreSQL + PostGIS |
| ORM/conexão | SQLAlchemy, psycopg2 |
| Clusterização | Scikit-learn (DBSCAN) |
| Machine Learning | Scikit-learn, XGBoost |
| Deep Learning (opcional) | Keras/TensorFlow, Prophet |
| Visualização | Plotly, Matplotlib, Seaborn |
| Mapas | Plotly Express (Mapbox), Folium |
| APIs climáticas | NASA POWER, Open-Meteo |
| Versionamento | Git + GitHub |
| Ambiente | Jupyter Notebook / Google Colab |


***

## Fase 10 — Critérios de Avaliação Atendidos

| Critério Zetta Lab | Como é Atendido |
| :-- | :-- |
| **Originalidade** | Combinação inédita de FMA + Angstrom + ML para severidade e previsão futura de incêndios no Brasil |
| **Inovação** | Clusterização por raio geoespacial + enriquecimento via NASA POWER automatizado |
| **Aderência à Agência Zetta** | Uso de base de dados real (INPE), análise exploratória, indicadores e visualização gráfica — exatamente o solicitado |
| **Organização do Trabalho** | Notebook estruturado em fases, código modular, banco de dados relacional, README |
| **Viabilidade Técnica** | Todas as ferramentas são open-source, APIs gratuitas, processamento em chunks viável |


***

> 💡 **Dica Final:** Para a apresentação, priorize o mapa de probabilidade de incêndio futuro com overlay nos biomas — é o visual mais impactante e diretamente ligado ao impacto social do projeto.
<span style="display:none">[^11][^14][^15][^2][^3][^4][^5][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://data.inpe.br/queimadas/dados-abertos/

[^2]: https://terrabrasilis.dpi.inpe.br/queimadas/situacao-atual/estatisticas/estatisticas_estados/

[^3]: https://basedosdados.org/dataset/f06f3cdc-b539-409b-b311-1ff8878fb8d9

[^4]: https://mapas.infraestruturameioambiente.sp.gov.br/server/rest/services/SIPAI/INPE_Focos_Queimadas_Acumulado/FeatureServer

[^5]: https://www.cnnbrasil.com.br/nacional/brasil-registrou-2783-mil-focos-de-incendio-em-2024-diz-inpe/

[^6]: https://www.youtube.com/watch?v=i1mPBITl8yA

[^7]: https://www.youtube.com/watch?v=LX0_ySodLds

[^8]: https://pt.scribd.com/document/959227865/2455-Texto-do-artigo-14950-17119-10-20230811

[^9]: https://www.gov.br/inpe/pt-br/acesso-a-informacao/dados-abertos

[^10]: https://data.inpe.br/queimadas/portal/faq/

[^11]: https://www.gov.br/inpe/pt-br/assuntos/ultimas-noticias/inpe-atualiza-painel-do-terrabrasilis-sobre-queimadas-desmatamento-e-tamanhos-de-propriedade

[^12]: https://data.inpe.br

[^13]: https://terrabrasilis.dpi.inpe.br/queimadas/bdqueimadas/

[^14]: https://mapas.infraestruturameioambiente.sp.gov.br/server/rest/services/SIPAI/INPE_Focos_Queimadas_Acumulado/MapServer

[^15]: https://terrabrasilis.dpi.inpe.br

