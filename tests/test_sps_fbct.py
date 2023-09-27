"""
Test script to properly read and fBCT file from SPS
"""
from ccc_miner import FBCT

# Load data into fBCT
parquet_file = 'test_data/FBCT_2023.09.22.16.07.26.612797.parquet'

# Instantiate class 
fbct = FBCT(parquet_file)
fbct.plot()