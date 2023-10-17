"""
Test script to properly read and fBCT file from SPS
"""
from pathlib import Path
from ccc_miner import FBCT

# Test data location 
data_folder = Path(__file__).resolve().parent.joinpath('../data').absolute()

# Load data into fBCT
parquet_file = '{}/test_data/FBCT_2023.09.22.16.07.26.612797.parquet'.format(data_folder)

# Instantiate class 
fbct = FBCT(parquet_file)
fbct.plot()