"""
Data Loader Module.
Responsible for ingesting raw data from CSV files or other sources.
"""
import pandas as pd
from typing import Optional
from pathlib import Path

def load_csv_data(filepath: Path) -> pd.DataFrame:
    """
    Loads traffic data from a CSV file.
    
    Args:
        filepath (Path): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    try:
        # ; as delimiter
        # error_bad_lines is deprecated, using on_bad_lines='skip' for pandas 2.0+ compatibility
        df = pd.read_csv(filepath, sep=';', on_bad_lines='skip')
        print(f"Successfully loaded data from {filepath} with shape {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def load_metadata(filepath: Path) -> pd.DataFrame:
    """
    Loads sensor metadata (coordinates, names).
    
    Args:
        filepath (Path): Path to the metadata CSV.
        
    Returns:
        pd.DataFrame: Metadata with columns [id, nombre, latitud, longitud]
    """
    try:
        # Metadata files often use Latin-1 (ISO-8859-1) encoding in Spain
        df = pd.read_csv(filepath, sep=';', on_bad_lines='skip', encoding='latin-1')
        
        # Clean column names (strip whitespace and lowercase immediately)
        df.columns = df.columns.str.strip().str.lower()
        
        # Map if necessary (sometimes 'codigo' instead of 'id')
        if 'id' not in df.columns and 'codigo' in df.columns:
            df.rename(columns={'codigo': 'id'}, inplace=True)
            
        # Ensure we have the columns we need
        # Expected: id, nombre, longitud, latitud
        needed_cols = ['id', 'nombre', 'longitud', 'latitud']
        
        if not all(col in df.columns for col in needed_cols):
             print(f"Metadata missing columns. Found: {df.columns.tolist()}")
             
        return df
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return pd.DataFrame()
