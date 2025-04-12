import sys
import os

# Print environment information for debugging
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")

try:
    import pandas as pd
    import numpy as np
    import json
    print(f"Successfully imported pandas version: {pd.__version__}")
    print(f"Successfully imported numpy version: {np.__version__}")
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("\nTroubleshooting steps:")
    print("1. Your virtual environment seems to have issues. Try creating a new one:")
    print("   - Exit current environment: deactivate")
    print("   - Remove existing environment: rm -rf env (Linux/Mac) or rmdir /s /q env (Windows)")
    print("   - Create new environment: python -m venv new_env")
    print("   - Activate new environment: .\\new_env\\Scripts\\activate (Windows) or source new_env/bin/activate (Linux/Mac)")
    print("   - Install dependencies: pip install -r requirements.txt")
    print("\n2. Or install packages directly to your system Python:")
    print("   - Deactivate virtual environment: deactivate")
    print("   - Install globally: pip install pandas numpy")
    print("\n3. If still having issues, try using conda instead of venv:")
    print("   - Install miniconda (if not already installed)")
    print("   - Create environment: conda create -n f1rag python=3.9")
    print("   - Activate: conda activate f1rag")
    print("   - Install packages: pip install -r requirements.txt")
    sys.exit(1)

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

def prepare_data_for_rag(race_info, output_file='data/processed/race_data.json'):
    """
    Prepare race information for the RAG model and save it to a file.
    
    Args:
        race_info (dict): Dictionary containing race information.
        output_file (str): Path to save the processed data.
        
    Returns:
        list: List of race data examples for the RAG model.
    """
    # Convert lap times to string representations for the model
    for driver, lap_times in race_info['lap_times'].items():
        race_info['lap_times'][driver] = [str(lt) for lt in lap_times]
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the processed data to a JSON file
    with open(output_file, 'w') as f:
        json.dump(race_info, f)
    
    # Create examples for the RAG model
    examples = []
    for driver in race_info['lap_times'].keys():
        example = f"Driver: {driver}\n"
        example += f"Lap times: {race_info['lap_times'].get(driver, [])}\n"
        example += f"Pit stops: {race_info['pit_stops'].get(driver, [])}\n"
        example += f"Overtakes: {race_info['overtakes'].get(driver, [])}\n"
        example += f"Incidents: {race_info['incidents'].get(driver, [])}\n"
        examples.append(example)
    
    return examples

if __name__ == "__main__":
    # Define file paths
    input_file = "data/raw/race_data.csv"
    output_file = "data/processed/race_data.json"
    examples_file = "data/processed/race_examples.json"
    
    print("Starting data processing...")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        print("Please make sure to place the race data CSV file in the data/raw directory.")
        exit(1)
    
    # Load the race data
    print(f"Loading race data from {input_file}...")
    df = load_race_data(input_file)
    
    # Preprocess the data
    print("Preprocessing race data...")
    df = preprocess_race_data(df)
    
    # Extract race information
    print("Extracting race information...")
    race_info = extract_race_information(df)
    
    # Prepare data for the RAG model
    print(f"Preparing data for the RAG model and saving to {output_file}...")
    examples = prepare_data_for_rag(race_info, output_file)
    
    # Save the examples
    os.makedirs(os.path.dirname(examples_file), exist_ok=True)
    with open(examples_file, 'w') as f:
        json.dump(examples, f)
    
    print(f"Data processing complete. Examples saved to {examples_file}.")
