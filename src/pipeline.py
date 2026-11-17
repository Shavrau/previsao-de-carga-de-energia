"""Orquestrador único do pipeline ponta a ponta (ingest -> transform -> baseline -> train -> evaluate)."""

from __future__ import annotations

import argparse
import subprocess
import sys


def _run_module(module: str, extra_args: list[str] | None = None) -> None:
    cmd = [sys.executable, "-m", module, *(extra_args or [])]
    print(f"[pipeline] Executando: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def run_pipeline(
    *,
    skip_ingest: bool = False,
    skip_mlflow: bool = False,
    skip_bonus: bool = False,
) -> None:
    """Executa todas as etapas obrigatórias e adicionais em sequência cronológica."""
    if not skip_ingest:
        _run_module("src.ingest")

    _run_module("src.transform")
    _run_module("src.baseline")

    mlflow_args = ["--skip-mlflow"] if skip_mlflow else []
    _run_module("src.train")
    _run_module("src.evaluate", mlflow_args)

    if not skip_bonus:
        _run_module("src.statistical", mlflow_args)
        _run_module("src.global_model", mlflow_args)

    print("[pipeline] Pipeline completo concluído com sucesso.", flush=True)


def main() -> None:
    """Ponto de entrada CLI do orquestrador de pipeline."""
    parser = argparse.ArgumentParser(
        description="Executa o pipeline completo de previsão de carga (ONS SE) com um único comando."
    )
    parser.add_argument("--skip-ingest", action="store_true", help="Reutiliza Bronze local sem baixar S3.")
    parser.add_argument("--skip-mlflow", action="store_true", help="Executa sem registrar runs no MLflow.")
    parser.add_argument(
        "--skip-bonus",
        action="store_true",
        help="Executa apenas as etapas obrigatórias 1..6 (pula AutoETS e modelo global).",
    )
    args = parser.parse_args()
    run_pipeline(
        skip_ingest=args.skip_ingest,
        skip_mlflow=args.skip_mlflow,
        skip_bonus=args.skip_bonus,
    )


if __name__ == "__main__":
    main()
