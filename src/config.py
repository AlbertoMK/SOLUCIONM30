"""
Global configurations for the Traffic Flow Optimization project.
Includes sensor locations, paths, and physics constants.
"""
import os
from pathlib import Path

# Project Root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Paths
DATA_PATH = BASE_DIR / "data"
DATA_PATH_RAW = DATA_PATH / "raw"
DATA_PATH_PROCESSED = DATA_PATH / "processed"
DATA_PATH_EXTERNAL = DATA_PATH / "external"

# M-30 Configuration
# List of dummy sensor IDs situated in the East Arc of M-30
M30_EAST_SENSORS = [
    1001,  # Sample ID found in data
    "PM-30-01", "PM-30-02", "PM-30-03", "PM-30-04", 
    "PM-30-05", "PM-30-06", "PM-30-07", "PM-30-08"
]

# Traffic Physics Constants
# Critical Density Threshold (percentage of occupancy or vehicles/km)
# Default set to 18% occupancy as a proxy for critical density
CRITICAL_DENSITY_THRESHOLD = 18.0 
