import pandas as pd
import numpy as np

def load_race_data(file_path):
    """
    Load race data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing race data.
        
    Returns:
        pd.DataFrame: DataFrame containing the race data.
    """
    return pd.read_csv(file_path)

def preprocess_race_data(df):
    """
    Preprocess race data by handling missing values and converting data types.
    
    Args:
        df (pd.DataFrame): DataFrame containing the race data.
        
    Returns:
        pd.DataFrame: Preprocessed race data.
    """
    # Handle missing values
    df.fillna(method='ffill', inplace=True)
    
    # Convert data types
    df['lap_time'] = pd.to_timedelta(df['lap_time'])
    
    return df

def extract_race_information(df):
    """
    Extract relevant race information such as lap times, pit stops, overtakes, and incidents.
    
    Args:
        df (pd.DataFrame): DataFrame containing the race data.
        
    Returns:
        dict: Dictionary containing extracted race information.
    """
    race_info = {
        'lap_times': df.groupby('driver')['lap_time'].apply(list).to_dict(),
        'pit_stops': df[df['event'] == 'pit_stop'].groupby('driver')['lap'].apply(list).to_dict(),
        'overtakes': df[df['event'] == 'overtake'].groupby('driver')['lap'].apply(list).to_dict(),
        'incidents': df[df['event'] == 'incident'].groupby('driver')['lap'].apply(list).to_dict()
    }
    
    return race_info
