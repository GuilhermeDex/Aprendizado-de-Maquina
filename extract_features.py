import pandas as pd
from pathlib import Path
from tqdm import tqdm
from detoxify import Detoxify
from typing import Iterable, List, Dict

DATA_PATH = Path("data/WELFake_Dataset.csv")
OUTPUT_PATH = Path("data/data_features.parquet")

TOXICITY_CATEGORIES = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
]

def lftk_extraction(df: pd.DataFrame, n_process: int = 6, batch_size: int = 500):
    import spacy
    import lftk
    import numpy as np
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor

    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    texts = df["text"].astype(str).tolist()
    feature_keys = None
    all_features = []

    def extract_features(doc):
        try:
            if len([t for t in doc if t.is_alpha]) == 0:
                return {key: np.nan for key in feature_keys}
            extractor = lftk.Extractor(docs=doc)
            extractor.customize(stop_words=True, punctuations=False, round_decimal=3)
            return extractor.extract()
        except Exception:
            return {key: np.nan for key in feature_keys}

    for start in tqdm(range(0, len(texts), batch_size), desc="Batch processing"):
        batch_texts = texts[start:start + batch_size]
        docs = list(nlp.pipe(batch_texts, n_process=n_process))

        if feature_keys is None:
            for doc in docs:
                if len([t for t in doc if t.is_alpha]) > 0:
                    try:
                        extractor = lftk.Extractor(docs=doc)
                        extractor.customize(stop_words=True, punctuations=False, round_decimal=3)
                        example_feats = extractor.extract()
                        feature_keys = example_feats.keys()
                        break
                    except Exception:
                        continue
            if feature_keys is None:
                raise ValueError("No valid doc for feature extraction.")

        with ThreadPoolExecutor(max_workers=n_process) as ex:
            batch_features = list(ex.map(extract_features, docs))

        all_features.extend(batch_features)

        del docs
        import gc; gc.collect()

    features_df = pd.DataFrame.from_records(all_features)
    return pd.concat([df.reset_index(drop=True), features_df], axis=1)


def ensure_dataset_exists(path: Path) -> None:
    """Ensure dataset file exists."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")


def init_columns(df: pd.DataFrame, columns: Iterable[str], value=None) -> None:
    """Initialize given columns with a default value."""
    for col in columns:
        df[col] = value

def assign_predictions(df: pd.DataFrame, results: List[Dict[str, float]], categories: List[str]) -> None:
    """Assign Detoxify results to the dataframe."""
    for category in categories:
        df[category] = [res.get(category, 0.0) for res in results]
        df[category] = pd.to_numeric(df[category], errors="coerce")

def main() -> None:
    ensure_dataset_exists(DATA_PATH)
    df = pd.read_csv(DATA_PATH, usecols=["text"])

    df = lftk_extraction(df, n_process=4)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Features saved on {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
