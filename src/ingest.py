"""Ingestão da camada Bronze a partir dos dados abertos do ONS no S3 via DuckDB httpfs."""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.request import urlopen

import duckdb

BUCKET_URL = "https://ons-aws-prod-opendata.s3.us-west-2.amazonaws.com"
DATASET_PREFIX = "dataset/carga_energia_di/"
BRONZE_DIR = Path("data/bronze")
START_YEAR = 2018
PARQUET_PATTERN = re.compile(r"^CARGA_ENERGIA_(\d{4})\.parquet$")
DICTIONARY_FILES = (
    "DicionarioDados_Carga_Energia_Diaria.pdf",
    "DicionarioDados_Carga_Energia_Diaria.json",
)
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def _sql_quote(value: str) -> str:
    """Escapa aspas simples para interpolação segura de literais em DDL/COPY do DuckDB."""
    return "'" + value.replace("'", "''") + "'"


def list_parquet_keys(start_year: int = START_YEAR) -> list[str]:
    """Lista no bucket público do ONS as chaves Parquet a partir de `start_year`."""
    keys: list[str] = []
    continuation_token: str | None = None

    while True:
        list_url = f"{BUCKET_URL}/?list-type=2&prefix={DATASET_PREFIX}"
        if continuation_token:
            list_url += f"&continuation-token={continuation_token}"

        with urlopen(list_url, timeout=30) as response:
            root = ET.fromstring(response.read())

        for content in root.findall("s3:Contents", S3_NS):
            key = content.findtext("s3:Key", default="", namespaces=S3_NS)
            filename = key.rsplit("/", 1)[-1]
            match = PARQUET_PATTERN.match(filename)
            if match and int(match.group(1)) >= start_year:
                keys.append(key)

        is_truncated = root.findtext("s3:IsTruncated", default="false", namespaces=S3_NS)
        if is_truncated.lower() != "true":
            break
        continuation_token = root.findtext(
            "s3:NextContinuationToken", default="", namespaces=S3_NS
        )
        if not continuation_token:
            break

    return sorted(keys)


def download_dictionaries(output_dir: Path) -> list[Path]:
    """Baixa os arquivos do dicionário de dados do ONS para consulta local."""
    saved: list[Path] = []
    for filename in DICTIONARY_FILES:
        url = f"{BUCKET_URL}/{DATASET_PREFIX}{filename}"
        target = output_dir / filename
        tmp_target = target.with_suffix(target.suffix + ".tmp")

        with urlopen(url, timeout=30) as response:
            tmp_target.write_bytes(response.read())

        tmp_target.replace(target)
        print(f"[bronze] {target.name}: {target.stat().st_size} bytes")
        saved.append(target)
    return saved


def ingest_bronze(
    output_dir: Path = BRONZE_DIR,
    start_year: int = START_YEAR,
    include_dictionaries: bool = True,
) -> list[Path]:
    """Lê os Parquets anuais do S3 via DuckDB httpfs e grava em `output_dir` sem alterações."""
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = list_parquet_keys(start_year=start_year)
    if not keys:
        raise RuntimeError(f"Nenhum arquivo Parquet encontrado a partir de {start_year}.")

    con = duckdb.connect()
    try:
        con.execute("INSTALL httpfs; LOAD httpfs;")
        saved_files: list[Path] = []

        for key in keys:
            url = f"{BUCKET_URL}/{key}"
            target = output_dir / Path(key).name
            tmp_target = target.with_name(target.name + ".tmp")

            con.execute(
                f"COPY (SELECT * FROM read_parquet({_sql_quote(url)})) "
                f"TO {_sql_quote(tmp_target.as_posix())} (FORMAT PARQUET)"
            )
            tmp_target.replace(target)

            row_count, min_dt, max_dt = con.execute(
                "SELECT COUNT(*), MIN(din_instante), MAX(din_instante) FROM read_parquet(?)",
                [target.as_posix()],
            ).fetchone()
            print(f"[bronze] {target.name}: rows={row_count}, min={min_dt}, max={max_dt}")
            saved_files.append(target)
    finally:
        con.close()

    if include_dictionaries:
        saved_files.extend(download_dictionaries(output_dir))

    return saved_files


def main() -> None:
    """Ponto de entrada CLI para ingestão da camada Bronze."""
    parser = argparse.ArgumentParser(
        description="Ingere dados diários de carga de energia do ONS (2018 até hoje) na camada Bronze."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BRONZE_DIR,
        help="Diretório de destino dos arquivos Bronze (padrão: data/bronze).",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=START_YEAR,
        help="Ano inicial dos arquivos a ingerir (padrão: 2018).",
    )
    parser.add_argument(
        "--skip-dictionaries",
        action="store_true",
        help="Não baixar os arquivos do dicionário de dados (PDF e JSON).",
    )
    args = parser.parse_args()

    ingest_bronze(
        output_dir=args.output_dir,
        start_year=args.start_year,
        include_dictionaries=not args.skip_dictionaries,
    )


if __name__ == "__main__":
    main()
