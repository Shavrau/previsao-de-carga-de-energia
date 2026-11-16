"""Orquestração em Python das transformações dbt + DuckDB (Silver e Gold)."""

from __future__ import annotations

import shutil
from pathlib import Path

from dbt.cli.main import dbtRunner

DBT_DIR = Path("dbt")
GOLD_DIR = Path("data/gold")
DOCS_OUT = Path("docs/dbt/index.html")


def run_dbt(*args: str) -> None:
    """Executa um subcomando dbt apontando para o projeto e profile locais."""
    result = dbtRunner().invoke(
        [
            *args,
            "--project-dir",
            str(DBT_DIR),
            "--profiles-dir",
            str(DBT_DIR),
        ]
    )
    if not result.success:
        raise RuntimeError(f"Falha ao executar 'dbt {' '.join(args)}'.")


def generate_docs() -> Path:
    """Gera a documentação estática do dbt e publica em `docs/dbt/index.html`."""
    run_dbt("docs", "generate", "--static")
    static_html = DBT_DIR / "target" / "static_index.html"
    DOCS_OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(static_html, DOCS_OUT)
    return DOCS_OUT


def main() -> None:
    """Garante o diretório de saída Gold e executa `dbt run`, `dbt test` e `dbt docs`."""
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    run_dbt("run")
    run_dbt("test")
    generate_docs()


if __name__ == "__main__":
    main()
