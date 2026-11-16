# Previsão de Carga de Energia (ONS - Subsistema SE)

Projeto de previsão da carga de energia diária (em MWmed) do subsistema Sudeste/Centro-Oeste (`SE`) para os próximos 7 dias, utilizando dados públicos do Operador Nacional do Sistema Elétrico (ONS).

## Pergunta do desafio

> O modelo prevê melhor do que simplesmente repetir o valor do mesmo dia da semana anterior? Se sim, quanto melhor?

_Em desenvolvimento — os resultados da comparação entre modelo e baseline serão documentados aqui ao final do pipeline._

## Etapas do projeto

- [x] **1. Ambiente**: Repositório inicializado com `pyproject.toml`, `.gitignore` e `README.md`
- [ ] **2. Ingestão (bronze)**: Dados de 2018 até hoje salvos em Parquet sem alterações
- [ ] **3. Transformação (silver e gold)**: Modelos dbt + DuckDB para a série diária do subsistema `SE` e testes
- [ ] **4. Baseline**: Previsão ingênua repetindo o valor do mesmo dia da semana anterior
- [ ] **5. Modelo**: LightGBM com lags e variáveis de calendário, registrado no MLflow
- [ ] **6. Avaliação**: Backtesting temporal comparando modelo e baseline em MAE e MAPE

## Como rodar

_Instruções serão adicionadas conforme as etapas do pipeline forem implementadas._
