import os
import re
import json
import pandas as pd
import html

try:
    import kagglehub
except ImportError as e:
    raise ImportError(
        "The 'kagglehub' module was not found. "
        "If you are running this outside Kaggle, "
        "please install: pip install kagglehub"
    ) from e

# --- CONFIGURATION ---
SAMPLES_PER_CLASS_TRAIN = 12500
SAMPLES_PER_CLASS_TEST = 1900
RANDOM_SEED = 42


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()

    # 1. Safely and completely unescape ALL HTML entities
    text = html.unescape(text)

    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # 3. Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def process_split(df, samples_per_class=None):
    df = df.copy()

    # Combine title and description
    df["text"] = (df["Title"].apply(clean_text) + " " +
                  df["Description"].apply(clean_text))

    # Shift labels: 1-4 -> 0-3
    df["label"] = df["Class Index"] - 1

    # Drop bad rows and duplicates
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text"])

    # Subsample if requested
    if samples_per_class:
        df = df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(
                n=min(samples_per_class, len(x)),
                random_state=RANDOM_SEED
            ),
            include_groups=False  # Fix Pandas FutureWarning
        ).reset_index(drop=True)

    return df[["text", "label"]]


def main():
    print("Downloading dataset...")
    path = kagglehub.dataset_download(
        "amananandrai/ag-news-classification-dataset"
    )

    print("Loading raw data...")
    train_raw = pd.read_csv(os.path.join(path, "train.csv"))
    test_raw = pd.read_csv(os.path.join(path, "test.csv"))

    print("Processing training data...")
    train_df = process_split(
        train_raw, samples_per_class=SAMPLES_PER_CLASS_TRAIN
    )

    print("Processing test data...")
    test_df = process_split(
        test_raw, samples_per_class=SAMPLES_PER_CLASS_TEST
    )

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    print("Saving processed data...")
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print("Saving id2label mapping...")
    id2label = {
        "0": "World", "1": "Sports",
        "2": "Business", "3": "Sci/Tech"
    }
    with open("id2label.json", "w") as f:
        json.dump(id2label, f, indent=2)

    print("Data preparation complete! Files saved to data/ directory.")


if __name__ == "__main__":
    main()
