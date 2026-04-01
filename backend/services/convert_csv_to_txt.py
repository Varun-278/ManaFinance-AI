import os
import pandas as pd

INPUT_FOLDER = "../data/NIFTY50"
OUTPUT_FOLDER = "../data/NIFTY50_TXT"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Columns required for price/stock history files
PRICE_REQUIRED = ["Date", "Open", "High", "Low", "Close"]
# Columns required for stock metadata file (optional)
META_REQUIRED = ["Company Name", "Industry", "Symbol"]


def is_price_file(df):
    """Detect if file is a price history CSV."""
    return all(col in df.columns for col in PRICE_REQUIRED)


def is_metadata_file(df):
    """Detect if file is stock_metadata.csv."""
    return all(col in df.columns for col in META_REQUIRED)


def price_row_to_text(row):
    """Convert OHLCV data into readable financial sentences."""
    def get(col):
        return row[col] if col in row and pd.notna(row[col]) else "N/A"

    symbol = get("Symbol")
    return (
        f"On {get('Date')}, stock {symbol} opened at {get('Open')}, "
        f"reached a high of {get('High')}, a low of {get('Low')}, "
        f"and closed at {get('Close')}. VWAP was {get('VWAP')}. "
        f"Total traded volume was {get('Volume')} shares, "
        f"turnover was {get('Turnover')}. Trades executed: {get('Trades')}. "
        f"Deliverable volume: {get('Deliverable Volume')} "
        f"with a deliverable percentage of {get('%Deliverble')}."
    )


def metadata_row_to_text(row):
    """Convert metadata into a clean descriptive sentence."""
    name = row["Company Name"]
    industry = row["Industry"]
    symbol = row["Symbol"]
    isin = row["ISIN Code"]

    return (
        f"{symbol} represents {name}, operating in the {industry} sector. "
        f"The ISIN code for this stock is {isin}."
    )


print("🔄 Starting CSV → TXT conversion...\n")

for filename in os.listdir(INPUT_FOLDER):
    if not filename.lower().endswith(".csv"):
        continue

    full_path = os.path.join(INPUT_FOLDER, filename)
    print(f"Processing {filename}...")

    df = pd.read_csv(full_path)

    # Case 1: Metadata file
    if is_metadata_file(df):
        print(f"📘 {filename} identified as METADATA file.")

        output_path = os.path.join(OUTPUT_FOLDER, "STOCK_METADATA.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                f.write(metadata_row_to_text(row) + "\n")

        print(f"✅ Converted metadata → STOCK_METADATA.txt\n")
        continue

    # Case 2: Price history file
    if is_price_file(df):
        print(f"💹 {filename} identified as PRICE DATA file.")

        df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"])

        output_path = os.path.join(OUTPUT_FOLDER, filename.replace(".csv", ".txt"))
        with open(output_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                f.write(price_row_to_text(row) + "\n")

        print(f"✅ Converted price file → {output_path}\n")
        continue

    # Unknown structure
    print(f"⚠️ Skipping {filename} — unrecognized format.\n")

print("🎉 Conversion complete! TXT files saved inside:", OUTPUT_FOLDER)
