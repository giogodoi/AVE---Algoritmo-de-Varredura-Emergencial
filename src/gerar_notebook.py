"""Script auxiliar para gerar o notebook analise_incendios.ipynb"""
import json, os

def code(src, id_):
    return {'cell_type':'code','execution_count':None,'id':id_,'metadata':{},'outputs':[],'source':src}

def md(src, id_):
    return {'cell_type':'markdown','id':id_,'metadata':{},'source':src}

cells = []

# ── TÍTULO ──────────────────────────────────────────────────────────────────
cells.append(md(
    "# 🔥 Análise de Incêndios no Brasil\n"
    "**Zetta Lab — Desafio III** | Previsão e Severidade de Focos de Incêndio\n\n"
    "Base: BDQueimadas/INPE — ~30M registros | PostgreSQL + PostGIS",
    'md-title'))

# ── 0. SETUP ─────────────────────────────────────────────────────────────────
cells.append(md("## 0. Setup & Imports", 'md-0'))

# Célula de instalação — garante que o kernel tenha todos os pacotes
cells.append(code(
    "import sys, subprocess\n"
    "pkgs = [\n"
    "    'matplotlib', 'seaborn', 'plotly', 'folium',\n"
    "    'scikit-learn', 'xgboost', 'joblib', 'pyarrow',\n"
    "    'sqlalchemy', 'psycopg2-binary', 'requests', 'tqdm'\n"
    "]\n"
    "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet'] + pkgs)\n"
    "print('Dependencias OK!')",
    'c-install'))

cells.append(code(
    "import sys, os\n"
    "sys.path.insert(0, os.path.join(os.getcwd(), '..', 'src'))\n\n"
    "import pandas as pd\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "import plotly.express as px\n"
    "import plotly.graph_objects as go\n"
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n\n"
    "from sqlalchemy import create_engine, text\n"
    "from data_loader import get_engine, load_sample_from_db, get_db_stats, save_parquet, load_parquet\n"
    "from features import (indice_angstrom, add_temporal_features, add_grid_cell,\n"
    "                      encode_bioma, classificar_frp, classificar_angstrom)\n"
    "from clustering import clusterizar_focos, resumo_clusters\n\n"
    "plt.rcParams['figure.figsize'] = (12, 5)\n"
    "sns.set_theme(style='whitegrid')\n"
    "print('Setup concluido!')",
    'c-setup'))


# ── 1. DADOS NO BANCO ────────────────────────────────────────────────────────
cells.append(md("## 1. Carregamento dos Dados", 'md-1'))
cells.append(code(
    "stats = get_db_stats()\n"
    "for k, v in stats.items():\n"
    "    print(f'{k:25s}: {v}')",
    'c-stats'))

cells.append(code(
    "PARQUET = '../csv/amostra_500k.parquet'\n\n"
    "if os.path.exists(PARQUET):\n"
    "    df = load_parquet(PARQUET)\n"
    "    print(f'Parquet carregado: {len(df):,} registros')\n"
    "else:\n"
    "    print('Carregando amostra do banco...')\n"
    "    df = load_sample_from_db(n=500_000)\n"
    "    save_parquet(df, PARQUET)\n"
    "    print(f'Amostra salva: {len(df):,} registros')\n\n"
    "df['data_hora'] = pd.to_datetime(df['data_hora'])\n"
    "df.head(3)",
    'c-load'))

# ── 2. LIMPEZA ───────────────────────────────────────────────────────────────
cells.append(md("## 2. Limpeza e Pré-processamento", 'md-2'))
cells.append(code(
    "print(f'Antes: {len(df):,} registros')\n"
    "print('Nulos:\\n', df.isnull().sum()[df.isnull().sum() > 0])\n\n"
    "df = df.replace(-999, np.nan).replace(-999.0, np.nan)\n"
    "df = df.dropna(subset=['latitude', 'longitude'])\n"
    "df = df.drop_duplicates(subset=['id'])\n\n"
    "print(f'Após limpeza: {len(df):,} registros')",
    'c-clean'))

cells.append(code(
    "# DBSCAN clustering por janela semanal (raio 100km)\n"
    "# Estratégia: clusterizar ANTES do enriquecimento para reduzir chamadas de API\n"
    "from clustering import clusterizar_focos, resumo_clusters\n\n"
    "print('Clusterizando focos por janela semanal (raio=100km)...')\n"
    "df['data_hora'] = pd.to_datetime(df['data_hora'])\n"
    "df['semana_cluster'] = df['data_hora'].dt.to_period('W').astype(str)\n\n"
    "resultados_clust = []\n"
    "for semana, grupo in df.groupby('semana_cluster'):\n"
    "    if len(grupo) < 2:\n"
    "        grupo['cluster_id'] = 0\n"
    "        resultados_clust.append(grupo)\n"
    "        continue\n"
    "    g_c = clusterizar_focos(grupo, raio_km=100)\n"
    "    g_c['semana_cluster'] = semana\n"
    "    resultados_clust.append(g_c)\n\n"
    "df_clust = pd.concat(resultados_clust, ignore_index=True)\n"
    "res = resumo_clusters(df, df_clust)\n"
    "print(f'Originais : {res[\"registros_originais\"]:,}')\n"
    "print(f'Clusters  : {res[\"registros_clusterizados\"]:,}')\n"
    "print(f'Reducao   : {res[\"reducao_pct\"]:.1f}% — pontos unicos para enriquecer via API')",
    'c-dbscan'))

# ── 3. SCHEMA POSTGRESQL (documentado) ──────────────────────────────────────
cells.append(md(
    "## 3. Schema PostgreSQL\n\n"
    "Tabela `focos_incendio` no banco `incendios_db` — PostgreSQL 18 + PostGIS 3.6.\n\n"
    "```sql\n"
    "CREATE TABLE focos_incendio (\n"
    "    id BIGINT PRIMARY KEY, data_hora TIMESTAMP,\n"
    "    satelite VARCHAR(30), pais VARCHAR(30), estado VARCHAR(50),\n"
    "    municipio VARCHAR(80), bioma VARCHAR(40),\n"
    "    dias_sem_chuva INTEGER, precipitacao DOUBLE PRECISION,\n"
    "    risco_fogo DOUBLE PRECISION, frp DOUBLE PRECISION,\n"
    "    latitude DOUBLE PRECISION, longitude DOUBLE PRECISION,\n"
    "    geom GEOMETRY(Point, 4326)\n"
    ");\n"
    "CREATE INDEX idx_focos_geom ON focos_incendio USING GIST(geom);\n"
    "```",
    'md-3'))

cells.append(code(
    "# Verificar conexão e contagem real no banco\n"
    "engine = get_engine()\n"
    "with engine.connect() as conn:\n"
    "    count = conn.execute(text('SELECT COUNT(*) FROM focos_incendio')).scalar()\n"
    "    print(f'Registros no banco: {count:,}')",
    'c-db-check'))

# ── 4. ENRIQUECIMENTO ────────────────────────────────────────────────────────
cells.append(md(
    "## 4. Enriquecimento com Dados Climáticos (NASA POWER API)\n\n"
    "**Fluxo otimizado:** Clusterizamos primeiro (Fase 2) reduzindo os pontos únicos,\n"
    "depois chamamos a NASA POWER API **apenas nos representantes de cluster**.\n"
    "Isso reduz as chamadas de potencialmente milhões para alguns milhares.\n\n"
    "A NASA POWER fornece dados históricos diários de temperatura (T2M) e umidade (RH2M)\n"
    "com resolução de 0.5° × 0.5° (~55km), gratuita e sem chave de API.",
    'md-4'))
cells.append(code(
    "from climate_api import enriquecer_representantes, propagar_clima_para_df\n\n"
    "# Os representantes de cluster já têm coordenadas únicas por região/semana\n"
    "# Limitar a 500 representantes nesta demonstração (cada chamada leva ~0.3s)\n"
    "LIMITE_DEMO = 500\n"
    "df_repr = df_clust.head(LIMITE_DEMO).copy()\n\n"
    "print(f'Enriquecendo {len(df_repr)} representantes via NASA POWER API...')\n"
    "print('(Em producao: rodar sobre todos os {len(df_clust):,} representantes)')\n\n"
    "df_repr_enriq = enriquecer_representantes(df_repr, delay_s=0.3, verbose=True)\n\n"
    "print('\\nAmostra enriquecida:')\n"
    "df_repr_enriq[['latitude','longitude','temperatura','umidade','frp']].head(5)",
    'c-enrich'))
cells.append(code(
    "# Propagar temperatura/umidade para o df completo via cluster_id\n"
    "# (todos os focos dentro do mesmo cluster 100km recebem o mesmo valor climático)\n"
    "if 'cluster_id' in df.columns and 'cluster_id' in df_repr_enriq.columns:\n"
    "    df = propagar_clima_para_df(df, df_repr_enriq, cluster_col='cluster_id')\n"
    "else:\n"
    "    # Fallback: merge direto pelos representantes enriquecidos\n"
    "    df = df.merge(\n"
    "        df_repr_enriq[['latitude','longitude','temperatura','umidade']].drop_duplicates(),\n"
    "        on=['latitude','longitude'], how='left'\n"
    "    )\n\n"
    "cobertura = df['temperatura'].notna().mean() * 100\n"
    "print(f'Cobertura climatica: {cobertura:.1f}% dos registros')\n"
    "print(f'Temperatura media  : {df[\"temperatura\"].mean():.1f} C')\n"
    "print(f'Umidade media      : {df[\"umidade\"].mean():.1f} %')",
    'c-enrich-propagar'))

# ── 5. FEATURE ENGINEERING ──────────────────────────────────────────────────
cells.append(md("## 5. Feature Engineering", 'md-5'))
cells.append(code(
    "df = add_temporal_features(df)\n"
    "df = add_grid_cell(df, resolucao=0.5)\n"
    "df['classe_frp'] = classificar_frp(df['frp'])\n\n"
    "# Angstrom (apenas onde há umidade/temperatura)\n"
    "mask = df['umidade'].notna() & df['temperatura'].notna()\n"
    "df.loc[mask, 'angstrom'] = indice_angstrom(df.loc[mask,'umidade'], df.loc[mask,'temperatura'])\n\n"
    "print('Distribuição classe_frp:')\n"
    "print(df['classe_frp'].value_counts())\n"
    "print('\\nFeatures geradas:', list(df.columns))",
    'c-feat'))

# ── 6. EDA ───────────────────────────────────────────────────────────────────
cells.append(md("## 6. Análise Exploratória (EDA)", 'md-6'))

cells.append(code(
    "# Focos por bioma\n"
    "fig = px.bar(\n"
    "    df['bioma'].value_counts().reset_index(),\n"
    "    x='bioma', y='count', title='Focos de Incêndio por Bioma',\n"
    "    color='count', color_continuous_scale='Oranges',\n"
    "    labels={'bioma':'Bioma','count':'Focos'}\n"
    ")\n"
    "fig.update_layout(showlegend=False)\n"
    "fig.show()",
    'c-eda-bioma'))

cells.append(code(
    "# Sazonalidade mensal\n"
    "por_mes = df.groupby('mes').size().reset_index(name='focos')\n"
    "fig = px.bar(por_mes, x='mes', y='focos',\n"
    "             title='Sazonalidade — Focos por Mês',\n"
    "             labels={'mes':'Mês','focos':'Focos'},\n"
    "             color='focos', color_continuous_scale='YlOrRd')\n"
    "fig.show()",
    'c-eda-mes'))

cells.append(code(
    "# Top 10 estados\n"
    "top_est = df['estado'].value_counts().head(10).reset_index()\n"
    "fig = px.bar(top_est, x='estado', y='count',\n"
    "             title='Top 10 Estados com Mais Focos',\n"
    "             color='count', color_continuous_scale='Reds')\n"
    "fig.show()",
    'c-eda-estados'))

cells.append(code(
    "# Mapa de densidade\n"
    "df_mapa = df.dropna(subset=['latitude','longitude']).sample(min(50_000, len(df)), random_state=42)\n"
    "fig = px.density_mapbox(\n"
    "    df_mapa, lat='latitude', lon='longitude', z='risco_fogo',\n"
    "    radius=4, zoom=3, center={'lat':-14,'lon':-52},\n"
    "    mapbox_style='carto-positron',\n"
    "    title='Mapa de Densidade — Focos de Incêndio',\n"
    "    color_continuous_scale='YlOrRd'\n"
    ")\n"
    "fig.show()",
    'c-eda-mapa'))

cells.append(code(
    "# Correlação entre variáveis numéricas\n"
    "cols_corr = ['dias_sem_chuva','precipitacao','risco_fogo','frp','mes']\n"
    "fig, ax = plt.subplots(figsize=(8, 6))\n"
    "sns.heatmap(df[cols_corr].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)\n"
    "ax.set_title('Correlação entre Variáveis')\n"
    "plt.tight_layout(); plt.show()",
    'c-eda-corr'))

# ── 7. MODELO SEVERIDADE ─────────────────────────────────────────────────────
cells.append(md("## 7. Modelo #1 — Previsão de Severidade (FRP)", 'md-7'))
cells.append(code(
    "from xgboost import XGBRegressor\n"
    "from sklearn.model_selection import train_test_split\n"
    "from sklearn.metrics import mean_absolute_error, r2_score\n"
    "from sklearn.preprocessing import LabelEncoder\n\n"
    "df_m = df.dropna(subset=['frp']).copy()\n"
    "le = LabelEncoder()\n"
    "df_m['bioma_enc']  = le.fit_transform(df_m['bioma'].fillna('Desconhecido'))\n"
    "df_m['estado_enc'] = le.fit_transform(df_m['estado'].fillna('Desconhecido'))\n\n"
    "FEATS = ['latitude','longitude','dias_sem_chuva','precipitacao',\n"
    "         'risco_fogo','mes','dia_do_ano','hora','bioma_enc','estado_enc']\n"
    "X = df_m[FEATS].fillna(-1)\n"
    "y = df_m['frp']\n\n"
    "X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)\n"
    "print(f'Treino: {len(X_tr):,} | Teste: {len(X_te):,}')",
    'c-m1-prep'))

cells.append(code(
    "modelo_sev = XGBRegressor(\n"
    "    n_estimators=300, max_depth=6, learning_rate=0.05,\n"
    "    subsample=0.8, colsample_bytree=0.8,\n"
    "    random_state=42, n_jobs=-1\n"
    ")\n"
    "modelo_sev.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)\n\n"
    "y_pred = modelo_sev.predict(X_te)\n"
    "print(f'MAE : {mean_absolute_error(y_te, y_pred):.2f} MW')\n"
    "print(f'R²  : {r2_score(y_te, y_pred):.4f}')",
    'c-m1-train'))

cells.append(code(
    "# Feature Importance\n"
    "fi = pd.Series(modelo_sev.feature_importances_, index=FEATS).sort_values()\n"
    "fig = px.bar(fi.reset_index(), x='feature_importances_', y='index',\n"
    "             orientation='h', title='Feature Importance — Severidade FRP',\n"
    "             color='feature_importances_', color_continuous_scale='Oranges',\n"
    "             labels={'index':'Feature','feature_importances_':'Importância'})\n"
    "fig.show()",
    'c-m1-fi'))

cells.append(code(
    "# Real vs Previsto\n"
    "fig = px.scatter(x=y_te[:5000], y=y_pred[:5000],\n"
    "                 labels={'x':'FRP Real (MW)','y':'FRP Previsto (MW)'},\n"
    "                 title='Severidade: Real vs. Previsto', opacity=0.4,\n"
    "                 color_discrete_sequence=['#FF6B35'])\n"
    "lim = float(y_te.max())\n"
    "fig.add_shape(type='line', x0=0, y0=0, x1=lim, y1=lim,\n"
    "              line=dict(color='black', dash='dash'))\n"
    "fig.show()",
    'c-m1-scatter'))

# ── 8. MODELO PREVISÃO FUTURA ────────────────────────────────────────────────
cells.append(md("## 8. Modelo #2 — Previsão de Ocorrência Futura", 'md-8'))
cells.append(code(
    "from xgboost import XGBClassifier\n"
    "from sklearn.metrics import classification_report, roc_auc_score\n\n"
    "# Grid 0.5° × 0.5° por semana\n"
    "df_grid = add_grid_cell(df.copy())\n"
    "df_grid['semana'] = df_grid['data_hora'].dt.to_period('W')\n\n"
    "focos_celula = (\n"
    "    df_grid.groupby(['cell_id','semana']).size()\n"
    "    .reset_index(name='n_focos')\n"
    ")\n"
    "focos_celula['incendio'] = (focos_celula['n_focos'] > 0).astype(int)\n\n"
    "stats_cel = df_grid.groupby('cell_id').agg(\n"
    "    lat_grid=('lat_grid','first'), lon_grid=('lon_grid','first'),\n"
    "    media_risco=('risco_fogo','mean'),\n"
    "    media_dias_seco=('dias_sem_chuva','mean'),\n"
    "    media_precip=('precipitacao','mean'),\n"
    "    n_historico=('id','count')\n"
    ").reset_index()\n\n"
    "df_clf = focos_celula.merge(stats_cel, on='cell_id')\n"
    "df_clf['mes'] = df_clf['semana'].dt.start_time.dt.month\n\n"
    "pos = df_clf['incendio'].sum()\n"
    "neg = (df_clf['incendio'] == 0).sum()\n"
    "print(f'Positivos: {pos:,} | Negativos: {neg:,}')",
    'c-m2-prep'))

cells.append(code(
    "FEATS_C = ['lat_grid','lon_grid','media_risco','media_dias_seco',\n"
    "           'media_precip','n_historico','mes']\n"
    "X_c = df_clf[FEATS_C].fillna(-1)\n"
    "y_c = df_clf['incendio']\n\n"
    "X_tc, X_vc, y_tc, y_vc = train_test_split(\n"
    "    X_c, y_c, test_size=0.2, random_state=42, stratify=y_c\n"
    ")\n\n"
    "modelo_prev = XGBClassifier(\n"
    "    n_estimators=300, max_depth=5, learning_rate=0.05,\n"
    "    scale_pos_weight=neg/pos if pos > 0 else 1,\n"
    "    eval_metric='logloss', random_state=42, n_jobs=-1\n"
    ")\n"
    "modelo_prev.fit(X_tc, y_tc, eval_set=[(X_vc, y_vc)], verbose=False)\n\n"
    "y_prob = modelo_prev.predict_proba(X_vc)[:,1]\n"
    "auc = roc_auc_score(y_vc, y_prob)\n"
    "print(f'AUC-ROC: {auc:.4f}')\n"
    "print(classification_report(y_vc, modelo_prev.predict(X_vc),\n"
    "      target_names=['Sem Incêndio','Com Incêndio']))",
    'c-m2-train'))

cells.append(code(
    "# Mapa de probabilidade futura\n"
    "df_clf['prob_incendio'] = modelo_prev.predict_proba(X_c)[:,1]\n\n"
    "fig = px.density_mapbox(\n"
    "    df_clf, lat='lat_grid', lon='lon_grid', z='prob_incendio',\n"
    "    radius=15, zoom=3, center={'lat':-14,'lon':-52},\n"
    "    mapbox_style='carto-positron',\n"
    "    color_continuous_scale='YlOrRd',\n"
    "    title='Probabilidade de Ocorrência de Incêndio por Célula Geoespacial'\n"
    ")\n"
    "fig.show()",
    'c-m2-mapa'))

cells.append(code(
    "# Curva ROC\n"
    "from sklearn.metrics import roc_curve\n"
    "fpr, tpr, _ = roc_curve(y_vc, y_prob)\n"
    "fig = go.Figure()\n"
    "fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',\n"
    "              name=f'ROC (AUC={auc:.3f})',\n"
    "              line=dict(color='#FF6B35', width=2)))\n"
    "fig.add_shape(type='line',x0=0,y0=0,x1=1,y1=1,\n"
    "              line=dict(dash='dash',color='gray'))\n"
    "fig.update_layout(title='Curva ROC — Previsão de Incêndio',\n"
    "                  xaxis_title='Taxa de Falsos Positivos',\n"
    "                  yaxis_title='Taxa de Verdadeiros Positivos')\n"
    "fig.show()",
    'c-m2-roc'))

# ── MONTAR E SALVAR ──────────────────────────────────────────────────────────
nb = {
    'nbformat': 4,
    'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.13.0'}
    },
    'cells': cells
}

os.makedirs('notebooks', exist_ok=True)
out = 'notebooks/analise_incendios.ipynb'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'[OK] Notebook criado: {out}')
print(f'   Celulas: {len(cells)}')
