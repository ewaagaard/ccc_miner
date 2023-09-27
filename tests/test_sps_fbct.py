"""
Test script to properly read and fBCT file from SPS
"""
from ccc_miner import SPS

# Instantiate class 
sps = SPS()

# Load data into fBCT
parquet_file = 'test_data/2023.09.22.16.07.26.612797.parquet'
sps.FBCT(parquet_file)