"""
matrix_builder.py
Constructs correlation matrices for cross-asset feature engineering.
"""
import os
import glob
import json
import polars as pl
import pandas as pd
import numpy as np
from typing import Dict, Any

class CorrelationMatrixBuilder:
    """
    Builds correlation matrices across NIFTY50 stocks to extract peer relationships.
    
    Cross-asset signals are crucial in HFT and quantitative trading because 
    macro-economic or sector-level information is often incorporated into different 
    stocks at slightly different speeds. If Stock A and Stock B are highly correlated, 
    a sudden price movement in Stock A might predict a subsequent movement in Stock B.
    
    Formulas:
    - Simple Return: R_t = (P_t - P_{t-1}) / P_{t-1}
      where P_t is the Mid-Price approximation: (open + close) / 2
    - Binary Direction: D_t = 1 if R_t > 0 else (-1 if R_t < 0 else 0)
    - Spike: S_t = 1 if abs(R_t) > 0.002 else 0
    - Pearson Correlation: cov(X,Y) / (sigma_X * sigma_Y)
    """
    
    def __init__(self, 
                 data_dir: str = r"d:\CODE\rajasthani\DATA\parquet", 
                 artifacts_dir: str = r"d:\CODE\rajasthani\trading_pipeline\artifacts"):
        """
        Initializes the CorrelationMatrixBuilder.
        
        Args:
            data_dir: Path to the directory containing NIFTY50 parquet files.
            artifacts_dir: Path to the directory where correlation matrices and maps will be saved.
        """
        self.data_dir = data_dir
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)
        
    def build(self) -> None:
        """
        Reads all Parquet files, computes 1-minute returns, aligns by timestamp, 
        and computes raw, binary, and spike correlation matrices. 
        Extracts the top 3 positive and negative peers for each stock and saves to disk.
        """
        file_paths = glob.glob(os.path.join(self.data_dir, "*.parquet"))
        
        if not file_paths:
            print("No parquet files found in the specified directory.")
            return
            
        dfs = []
        for fp in file_paths:
            symbol = os.path.basename(fp).replace(".parquet", "")
            
            # Lazy scan for memory efficiency
            # Compute mid-price = (open + close) / 2
            # Compute return = mid_price / shifted(mid_price) - 1
            df = (
                pl.scan_parquet(fp)
                .select([
                    pl.col("date").alias("timestamp"),
                    pl.col("open"),
                    pl.col("high"),
                    pl.col("low"),
                    pl.col("close")
                ])
                .with_columns(
                    ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("mid_price")
                )
                .sort("timestamp")
                .with_columns(
                    (pl.col("mid_price") / pl.col("mid_price").shift(1) - 1.0).alias(symbol)
                )
                .select(["timestamp", symbol])
            )
            dfs.append(df)
            
        # Join all lazy frames on timestamp
        # For memory efficiency with Polars, we collect the first dataframe, 
        # and iteratively outer-join the others.
        master_df = dfs[0].collect()
        for df in dfs[1:]:
            master_df = master_df.join(df.collect(), on="timestamp", how="outer_coalesce")
            
        # Drop timestamp for correlation matrix computation and fill nulls with 0
        returns_df = master_df.drop("timestamp").fill_null(0.0)
        
        # Convert to Pandas for computing the NxN correlation matrix
        pdf = returns_df.to_pandas()
        
        # a) Raw return correlations (Pearson)
        raw_corr = pdf.corr(method="pearson")
        
        # b) Binary direction correlations (+1, -1, 0)
        binary_pdf = np.sign(pdf)
        bin_corr = binary_pdf.corr(method="pearson")
        
        # c) 0.2% spike correlations (binary 1 if abs(R_t) > 0.002 else 0)
        spike_pdf = (pdf.abs() > 0.002).astype(float)
        spike_corr = spike_pdf.corr(method="pearson")
        
        # Save raw correlation matrix to artifacts directory
        # Save all 3 correlation matrices to artifacts directory
        raw_corr_path = os.path.join(self.artifacts_dir, "correlation_matrix.parquet")
        raw_corr.to_parquet(raw_corr_path)
        pdf.corr(method="pearson").to_parquet(os.path.join(self.artifacts_dir, "correlation_matrix_raw.parquet"))
        bin_corr.to_parquet(os.path.join(self.artifacts_dir, "correlation_matrix_binary.parquet"))
        spike_corr.to_parquet(os.path.join(self.artifacts_dir, "correlation_matrix_spike.parquet"))
        
        # Extract top 3 positive and top 3 negative peers for each symbol
        def _extract_peers(corr_matrix, symbols):
            peers = {}
            for sym in symbols:
                sym_corr = corr_matrix[sym].drop(sym)
                peers[sym] = {
                    "top_positive": sym_corr.nlargest(3).index.tolist(),
                    "top_negative": sym_corr.nsmallest(3).index.tolist()
                }
            return peers

        symbols = raw_corr.columns.tolist()
        raw_peers = _extract_peers(raw_corr, symbols)
        binary_peers = _extract_peers(bin_corr, symbols)
        spike_peers = _extract_peers(spike_corr, symbols)
        
        peer_map: Dict[str, Dict[str, Any]] = {}
        
        for sym in symbols:
            peer_map[sym] = {
                "correlated_peers_raw": raw_peers[sym],
                "correlated_peers_binary": binary_peers[sym],
                "correlated_peers_spike": spike_peers[sym]
            }
            
        # Save peer map to JSON
        peer_map_path = os.path.join(self.artifacts_dir, "peer_map.json")
        with open(peer_map_path, "w") as f:
            json.dump(peer_map, f, indent=4)
            
        print("Correlation matrices and peer map generated successfully.")

if __name__ == '__main__':
    builder = CorrelationMatrixBuilder()
    builder.build()

