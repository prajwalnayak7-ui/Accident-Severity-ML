import pandas as pd
from pathlib import Path

# define paths
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
INTERIM_DIR = Path(__file__).resolve().parents[1] / "data" / "interim"
RAW_FILE = RAW_DIR / "US_Accidents_March23.csv"

def load_data():
    print(f"Loading dataset from: {RAW_FILE}")
    df = pd.read_csv(RAW_FILE)
    print("✅ Data loaded successfully!")
    print(f"Shape: {df.shape}")
    print("\nColumns:\n", df.columns.tolist()[:20])  # show first 20 columns
    return df

def clean_and_sample(df):
    # remove rows with missing target
    if "Severity" in df.columns:
        df = df.dropna(subset=["Severity"])
    else:
        print("⚠️ Column 'Severity' not found. Please verify your CSV columns.")
    
    # optional: create smaller sample for testing (e.g., 10,000 rows)
    sample_df = df.sample(n=10000, random_state=42)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(INTERIM_DIR / "sample_accidents.csv", index=False)
    print("✅ Sample saved to data/interim/sample_accidents.csv")
    return sample_df

if __name__ == "__main__":
    df = load_data()
    sample_df = clean_and_sample(df)
    print("\n✅ Cleaning complete! Next: we’ll train a model on this sample dataset.")
