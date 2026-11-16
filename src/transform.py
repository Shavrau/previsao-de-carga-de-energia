"""Orquestração em Python das transformações dbt + DuckDB (Silver e Gold)."""

from __future__ import annotations

from pathlib import Path

from dbt.cli.main import dbtRunner

DBT_DIR = Path("dbt")
GOLD_DIR = Path("data/gold")


def run_dbt(command: str) -> None:
    """Executa um subcomando dbt apontando para o projeto e profile locais."""
    result = dbtRunner().invoke(
        [
            command,
            "--project-dir",
            str(DBT_DIR),
            "--profiles-dir",
            str(DBT_DIR),
        ]
    )
    if not result.success:
        raise RuntimeError(f"Falha ao executar 'dbt {command}'.")


def main() -> None:
    """Garante o diretório de saída Gold e executa `dbt run` e `dbt test`."""
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    run_dbt("run")
    run_dbt("test")


if __name__ == "__main__":
    main()
