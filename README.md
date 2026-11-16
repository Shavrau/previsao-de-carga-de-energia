# Previsão de Carga de Energia (ONS - Subsistema SE)

Projeto de previsão da carga de energia diária (em MWmed) do subsistema Sudeste/Centro-Oeste (`SE`) para os próximos 7 dias, utilizando dados públicos do Operador Nacional do Sistema Elétrico (ONS).

## Pergunta do desafio

> O seu modelo prevê melhor do que simplesmente repetir o valor do mesmo dia da semana anterior? Se sim, quanto melhor?

**Sim.** No backtesting temporal com janela expansiva de **52 semanas (`364` dias, de 16/11/2025 a 14/11/2026)**, o modelo **LightGBM (`full_history__regression_l1`)** alcançou **MAE de `1.730,58` MWmed** e **MAPE de `3,891%`**, superando o baseline ingênuo (`D-7`: **MAE `2.322,70` MWmed** e **MAPE `5,225%`**). Isso representa uma **redução de `25,49%` no MAE** (`-592,12` MWmed e `-1,334 p.p.` no MAPE), com ganho ainda mais expressivo em feriados, vésperas, pós-feriados e ecos de `D-7` (**`-52,32%` no MAE**: `1.663,71` vs `3.489,13` MWmed).

## Resultados do Backtesting (52 semanas)

| Modelo | Dias | MAE (MWmed) | MAPE (%) | RMSE (MWmed) | Ganho MAE vs `D-7` |
|---|---:|---:|---:|---:|---:|
| **`LightGBM (full_history__regression_l1)`** | 364 | **1.730,58** | **3,891%** | **2.214,53** | **+25,49%** |
| `LightGBM (post_mmgd__regression)` | 364 | 1.780,97 | 4,001% | 2.280,39 | +23,32% |
| `Baseline (D-7)` | 364 | 2.322,70 | 5,225% | 2.893,26 | 0,00% |

| Tipo de Dia | Dias | MAE `Baseline D-7` | MAE `LightGBM` | Ganho no MAE |
|---|---:|---:|---:|---:|
| Feriado / Véspera / Pós / Eco `D-7` | 38 | 3.489,13 | **1.663,71** | **+52,32%** |
| Dia Comum | 326 | 2.186,74 | **1.738,37** | **+20,50%** |

![Série Real vs Baseline D-7 vs LightGBM e Previsão 7 Dias](docs/figures/backtest_series_and_forecast.png)

![Comparação de MAE por Dia da Semana e Tipo de Dia](docs/figures/error_comparison_breakdown.png)

## Etapas do projeto

- [x] **1. Ambiente**: Repositório inicializado com `pyproject.toml`, `.gitignore`, `uv` e `Docker`
- [x] **2. Ingestão (bronze)**: Dados de 2018 até hoje salvos em `data/bronze/*.parquet` via DuckDB `httpfs` sem alterações
- [x] **3. Transformação (silver e gold)**: Modelos dbt + DuckDB para a série diária do subsistema `SE` e **17 testes** passando
- [x] **4. Baseline**: Previsão ingênua `D-7` em `pandas` (`src/baseline.py`)
- [x] **5. Modelo**: `LightGBM` com lags $\ge 7$ dias, calendário e feriados BR (`src/features.py` e `src/train.py`), registrado no `MLflow` (Docker)
- [x] **6. Avaliação**: Backtesting temporal de 52 semanas comparando modelo e baseline em MAE e MAPE (`src/evaluate.py`)
- [x] **Bônus**: Dashboard interativo em **Streamlit** (`src/app.py`), **Docker Compose** e testes automatizados com **pytest**

## Como rodar

### 1. Instalar dependências
```bash
uv sync
```

### 2. Ingestão (camada Bronze)
Baixa os arquivos Parquet anuais de carga diária do ONS (2018 até hoje) diretamente do bucket público S3 (`us-west-2`) via DuckDB `httpfs` para `data/bronze/`, junto com o dicionário de dados:
```bash
uv run python -m src.ingest
```

### 3. Transformação (camadas Silver e Gold)
Executa os modelos, testes e geração de documentação estática do `dbt` sobre o DuckDB (`data/warehouse.duckdb`), gerando `silver.silver_carga_energia_diaria`, `gold.gold_carga_energia_se`, `data/gold/carga_energia_se.parquet` e `docs/dbt/index.html`:
```bash
uv run python -m src.transform
```

### 4. Baseline (`D-7`)
Gera o histórico alinhado e a previsão dos próximos 7 dias repetindo o valor do mesmo dia da semana anterior (`data/predictions/baseline_d7.parquet`):
```bash
uv run python -m src.baseline
```

### 5. Modelo (`LightGBM + MLflow` com Docker)
Suba o servidor MLflow em Docker (no WSL2) e execute a grade de experimentos + previsão dos próximos 7 dias (`data/predictions/lgbm_forecast_7d.parquet`):
```bash
wsl -e bash -lc "cd /mnt/c/previsso_de_carga_de_energia && docker compose up -d --wait mlflow"
uv run python -m src.train
```
Ou execute o servidor MLflow e o treinamento inteiramente em containers:
```bash
wsl -e bash -lc "cd /mnt/c/previsso_de_carga_de_energia && docker compose up -d --wait mlflow && docker compose --profile pipeline run --rm --build pipeline"
```
A interface do MLflow fica acessível em `http://localhost:5000` enquanto o container estiver ativo (caso contrário, `src.train` utiliza fallback automático para `sqlite:///mlruns/mlflow.db`).

### 6. Avaliação (Backtesting 52 semanas) e Dashboard Streamlit
```bash
uv run python -m src.evaluate
uv run streamlit run src/app.py
```

### Testes unitários
```bash
uv run pytest -v
```

## O que faria com mais tempo

- **Variáveis meteorológicas exógenas (INMET):** incorporar temperatura e sensação térmica previstas para as principais capitais do `SE` (SP, RJ, MG), pois ondas de calor e frentes frias explicam boa parte dos picos residuais em dias comuns.
- **Segundo concorrente estatístico (`statsforecast`):** comparar o `LightGBM` contra `AutoETS` e `AutoARIMA` ou combinar via ensemble.
- **Modelo global multi-subsistema:** treinar um modelo único com `SE`, `S`, `NE` e `N` compartilhando dinâmica semanal e efeito de feriados nacionais/regionais.
- **Orquestração e qualidade:** automatizar o fluxo completo em um único alvo no `Makefile` / Docker Compose e incluir `ruff` no CI do GitHub Actions.
