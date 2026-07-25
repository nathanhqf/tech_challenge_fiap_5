"""Prepara dataset Bank Marketing para modelagem.

Carrega bank-additional-full.csv, descarta colunas com vazamento temporal
(duration) e salva tabela de modelagem em Parquet.
"""
import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RAW_PATH = os.environ.get(
    "KAGGLE_CSV_PATH",
    str(Path(__file__).parent.parent / "data" / "kaggle" / "bank-additional-full.csv"),
)
OUT_PATH = "data/processed/modeling_table.parquet"

# Colunas descartadas por vazamento temporal ou privacidade
DROP_COLS = ["duration"]  # só conhecida após o contato — vazamento temporal


def prepare(raw_path: str = RAW_PATH, out_path: str = OUT_PATH) -> pd.DataFrame:
    """Carrega, limpa e salva dataset para modelagem.

    Args:
        raw_path: Caminho do CSV original do Kaggle.
        out_path: Destino do Parquet processado.

    Returns:
        DataFrame processado.
    """
    logger.info("Carregando: %s", raw_path)
    df = pd.read_csv(raw_path, sep=";")
    logger.info("Shape original: %s", df.shape)

    # Remove vazamento temporal
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    logger.info("Colunas após descarte de duration: %s", list(df.columns))

    # Codifica target
    df["y_binary"] = (df["y"] == "yes").astype(int)

    # Features derivadas para contexto do bandit
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 30, 40, 50, 60, 100],
        labels=["<30", "30-40", "40-50", "50-60", "60+"],
        right=True,
    ).astype(str)

    df["previous_contact"] = (df["previous"] > 0).astype(int)

    job_map = {
        "management": "management", "admin.": "admin", "technician": "technician",
        "blue-collar": "blue_collar", "services": "services", "retired": "retired",
        "entrepreneur": "entrepreneur", "self-employed": "self_employed",
        "student": "student", "housemaid": "housemaid", "unemployed": "unemployed",
        "unknown": "unknown",
    }
    df["job_group"] = df["job"].map(job_map).fillna("unknown")

    segment_map = {"success": "recorrente", "failure": "reativado", "nonexistent": "novo"}
    df["segment"] = df["poutcome"].map(segment_map).fillna("novo")

    channel_map = {"cellular": "app", "telephone": "web"}
    df["channel"] = df["contact"].map(channel_map).fillna("web")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Salvo em %s | shape=%s | conversão=%.1f%%",
                out_path, df.shape, df["y_binary"].mean() * 100)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = prepare()
    print(df[["age_group", "job_group", "segment", "channel", "y_binary"]].head(10))
    print("\nDistribuição do target:")
    print(df["y_binary"].value_counts(normalize=True).round(3))
    print("\nDistribuição de segmentos:")
    print(df["segment"].value_counts())
