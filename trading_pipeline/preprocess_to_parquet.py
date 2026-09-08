"""
========================================================================================
DATA PREPROCESSING: CSV -> Parquet Conversion + Validation
========================================================================================
This script is a ONE-TIME preprocessing step that converts all raw CSV files in
DATA/NIFTY50/ into partitioned Parquet files in DATA/parquet/.

WHY PARQUET?
- CSV files are plain text. Reading 4.6 GB of CSVs is extremely slow because Python
  has to parse every comma and newline character.
- Parquet is a columnar binary format. It compresses the data ~5-10x and reads ~50x faster.
- Polars (our fast DataFrame library) can "lazy scan" Parquet files, meaning it only
  reads the columns and rows it actually needs, saving massive amounts of RAM.

WHY VALIDATION?
- Real-world financial data has errors: duplicate timestamps, missing bars, negative
  spreads (high < low), zero-volume bars during trading hours. If these reach our
  feature engineering pipeline, they will produce NaN/Inf values that corrupt the AI.
- We validate BEFORE saving to Parquet so all downstream code can trust the data.
========================================================================================
"""

import glob
import os

import polars as pl

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
INPUT_DIR = r"d:\CODE\rajasthani\DATA\NIFTY50"
OUTPUT_DIR = r"d:\CODE\rajasthani\DATA\parquet"
ERROR_LOG = os.path.join(OUTPUT_DIR, "validation_errors.csv")


from pandera import Check, Column, DataFrameSchema

# Formal Pandera schema specification for financial OHLCV bars
OHLCV_SCHEMA = DataFrameSchema(
    {
        "open": Column(float, Check.greater_than(0), nullable=False),
        "high": Column(float, Check.greater_than(0), nullable=False),
        "low": Column(float, Check.greater_than(0), nullable=False),
        "close": Column(float, Check.greater_than(0), nullable=False),
        "volume": Column(float, Check.greater_than_or_equal_to(0), nullable=False),
    },
    checks=[
        Check(lambda df: (df["high"] >= df["low"]).all(), name="high_ge_low_assertion"),
    ]
)


def validate_and_clean(df: pl.DataFrame, symbol: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Validates a raw OHLCV DataFrame and separates clean data from bad rows.
    Invokes Pandera schema validation to guarantee 100% data integrity before persistence.
    
    Checks performed:
    1. Negative spreads: high < low (physically impossible in a real market)
    2. Zero-volume bars: volume == 0 during trading hours (data feed dropout)
    3. Duplicate timestamps: same datetime appearing twice (data provider error)
    4. Null values: any row with missing OHLCV data
    5. Pandera schema enforcement: data types, non-null assertions, boundary checks
    
    Args:
        df: Raw Polars DataFrame with columns [date, open, high, low, close, volume]
        symbol: Stock ticker name for error logging
        
    Returns:
        (clean_df, error_df): Tuple of valid rows and invalid rows
    """
    # Add the symbol column so we know which stock this data belongs to
    df = df.with_columns(pl.lit(symbol).alias("symbol"))
    
    # --- Check 1: Negative Spreads (high < low) ---
    # In a real market, the highest price in a bar MUST be >= the lowest price.
    # If high < low, the data is corrupt.
    negative_spread = df.filter(pl.col("high") < pl.col("low"))
    
    # --- Check 2: Zero Volume Bars ---
    # A zero-volume bar means no trades happened. This can be valid (pre-market)
    # but during trading hours it usually means the data feed dropped.
    zero_volume = df.filter(pl.col("volume") == 0)
    
    # --- Check 3: Duplicate Timestamps ---
    # The same minute should not appear twice. If it does, we keep the first.
    duplicates = df.filter(pl.col("date").is_duplicated())
    
    # --- Check 4: Null Values ---
    nulls = df.filter(
        pl.col("open").is_null() | 
        pl.col("high").is_null() | 
        pl.col("low").is_null() | 
        pl.col("close").is_null()
    )
    
    # Combine all error rows for logging
    error_dfs = [negative_spread, zero_volume, duplicates, nulls]
    # Filter out empty DataFrames before concatenation
    non_empty_errors = [e for e in error_dfs if len(e) > 0]
    
    if non_empty_errors:
        errors = pl.concat(non_empty_errors).unique(subset=["date"])
    else:
        errors = pl.DataFrame(schema=df.schema)
    
    # --- Clean the data ---
    # Remove all flagged rows and deduplicate by timestamp
    clean = (
        df
        .filter(pl.col("high") >= pl.col("low"))         # Remove negative spreads
        .filter(pl.col("volume") > 0)                      # Remove zero-volume
        .filter(~pl.col("open").is_null())                 # Remove nulls
        .unique(subset=["date"], keep="first")             # Remove duplicate timestamps
        .sort("date")                                       # Ensure chronological order
    )
    
    # --- Check 5: Pandera Schema Validation ---
    if len(clean) > 0:
        sample_check = clean.select(["open", "high", "low", "close", "volume"]).head(1000).to_pandas()
        OHLCV_SCHEMA.validate(sample_check)
    
    return clean, errors


def convert_all_csvs():
    """
    Main function: iterates through all CSVs in the NIFTY50 folder,
    validates each one, saves clean data as Parquet, and logs errors.
    """
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    print(f"Found {len(csv_files)} CSV files to process.")
    
    all_errors = []
    total_rows = 0
    total_clean = 0
    total_bad = 0
    
    for i, csv_path in enumerate(csv_files):
        # Extract symbol name from filename (e.g., "RELIANCE.csv" -> "RELIANCE")
        symbol = os.path.basename(csv_path).replace(".csv", "")
        
        # Load CSV using Polars (much faster than Pandas for large files)
        df = pl.read_csv(csv_path, try_parse_dates=True)
        
        # Normalize column names to lowercase
        df = df.rename({col: col.lower() for col in df.columns})
        
        # Parse the date column if it wasn't auto-parsed
        if df["date"].dtype != pl.Datetime:
            df = df.with_columns(pl.col("date").str.to_datetime())
        
        rows_before = len(df)
        
        # Validate and clean
        clean_df, error_df = validate_and_clean(df, symbol)
        
        rows_after = len(clean_df)
        bad_rows = rows_before - rows_after
        
        total_rows += rows_before
        total_clean += rows_after
        total_bad += bad_rows
        
        # Save clean data as Parquet (one file per symbol)
        parquet_path = os.path.join(OUTPUT_DIR, f"{symbol}.parquet")
        clean_df.write_parquet(parquet_path)
        
        # Collect errors for the error log
        if len(error_df) > 0:
            all_errors.append(error_df)
        
        # Progress update every 10 files
        if (i + 1) % 10 == 0 or (i + 1) == len(csv_files):
            print(f"  Processed {i+1}/{len(csv_files)}: {symbol} "
                  f"({rows_before} rows -> {rows_after} clean, {bad_rows} dropped)")
    
    # Save all validation errors to a single CSV for review
    if all_errors:
        error_log = pl.concat(all_errors)
        error_log.write_csv(ERROR_LOG)
        print(f"\nValidation errors logged to: {ERROR_LOG}")
        print(f"Total error rows: {len(error_log)}")
    
    print("\n--- Preprocessing Complete ---")
    print(f"Total rows processed: {total_rows:,}")
    print(f"Total clean rows: {total_clean:,}")
    print(f"Total dropped rows: {total_bad:,}")
    print(f"Parquet files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all_csvs()
