# Previsão de Carga de Energia (ONS - Subsistema SE)

Projeto de previsão da carga de energia diária (em MWmed) do subsistema Sudeste/Centro-Oeste (`SE`) para os próximos 7 dias, utilizando dados públicos do Operador Nacional do Sistema Elétrico (ONS).

## Pergunta do desafio

> O modelo prevê melhor do que simplesmente repetir o valor do mesmo dia da semana anterior? Se sim, quanto melhor?

_Em desenvolvimento — os resultados da comparação entre modelo e baseline serão documentados aqui ao final do pipeline._

## Etapas do projeto

- [x] **1. Ambiente**: Repositório inicializado com `pyproject.toml`, `.gitignore` e `README.md`
- [x] **2. Ingestão (bronze)**: Dados de 2018 até hoje salvos em Parquet sem alterações
- [x] **3. Transformação (silver e gold)**: Modelos dbt + DuckDB para a série diária do subsistema `SE` e testes
- [ ] **4. Baseline**: Previsão ingênua repetindo o valor do mesmo dia da semana anterior
- [ ] **5. Modelo**: LightGBM com lags e variáveis de calendário, registrado no MLflow
- [ ] **6. Avaliação**: Backtesting temporal comparando modelo e baseline em MAE e MAPE

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
Executa os modelos, testes e geração de documentação do `dbt` sobre o DuckDB (`data/warehouse.duckdb`), gerando `silver.silver_carga_energia_diaria`, `gold.gold_carga_energia_se`, exportando `data/gold/carga_energia_se.parquet` e atualizando `docs/dbt/index.html`:
```bash
uv run python -m src.transform
```
Ou diretamente via CLI do dbt:
```bash
uv run dbt run --project-dir dbt --profiles-dir dbt
uv run dbt test --project-dir dbt --profiles-dir dbt
uv run dbt docs generate --project-dir dbt --profiles-dir dbt --static
uv run dbt docs serve --project-dir dbt --profiles-dir dbt
```
