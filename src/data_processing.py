"""
Data processing script for Formula 1 RAG system.
Processes raw CSV files from the Formula 1 dataset (1950-2020) from Kaggle.
"""
import os
import json
import argparse
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data_processing.log")
    ]
)
logger = logging.getLogger(__name__)

def load_csv_data(archive_dir):
    """
    Load CSV data from the archive directory
    
    Args:
        archive_dir (str): Directory containing the CSV files
        
    Returns:
        dict: Dictionary of DataFrames with file names as keys
    """
    logger.info(f"Loading CSV data from {archive_dir}")
    
    csv_files = [f for f in os.listdir(archive_dir) if f.endswith('.csv')]
    
    if not csv_files:
        logger.error(f"No CSV files found in {archive_dir}")
        return {}
    
    data = {}
    for file in csv_files:
        try:
            file_path = os.path.join(archive_dir, file)
            df = pd.read_csv(file_path)
            data[file.replace('.csv', '')] = df
            logger.info(f"Loaded {file} with {len(df)} rows")
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")
    
    return data

def extract_race_information(data_dict):
    """
    Extract race information from the data
    
    Args:
        data_dict (dict): Dictionary of DataFrames
        
    Returns:
        list: List of dictionaries with race information
    """
    logger.info("Extracting race information")
    
    races_info = []
    
    # Check if we have the necessary files
    required_files = ['races', 'circuits', 'results', 'drivers', 'lap_times']
    missing_files = [f for f in required_files if f not in data_dict]
    
    if missing_files:
        logger.warning(f"Missing required data files: {missing_files}")
        
    # If we have races and circuits data, we can extract basic race information
    if 'races' in data_dict and 'circuits' in data_dict:
        races_df = data_dict['races']
        circuits_df = data_dict['circuits']
        
        # Join races with circuits
        try:
            races_with_circuits = pd.merge(
                races_df,
                circuits_df,
                left_on='circuitId',
                right_on='circuitId',
                how='left'
            )
            
            # Process each race
            for _, race in races_with_circuits.iterrows():
                race_info = {
                    'race_id': int(race['raceId']),
                    'race_name': race['name_x'],
                    'circuit_name': race['name_y'],
                    'circuit_location': f"{race['location']}, {race['country']}",
                    'date': race['date'],
                    'year': int(race['year']),
                    'round': int(race['round']),
                    'lap_times': {}
                }
                
                # Add lap times if available
                if 'lap_times' in data_dict:
                    lap_times_df = data_dict['lap_times']
                    race_laps = lap_times_df[lap_times_df['raceId'] == race_info['race_id']]
                    
                    if not race_laps.empty:
                        # Add driver information if available
                        if 'drivers' in data_dict:
                            drivers_df = data_dict['drivers']
                            for driver_id in race_laps['driverId'].unique():
                                driver_info = drivers_df[drivers_df['driverId'] == driver_id]
                                if not driver_info.empty:
                                    driver_name = f"{driver_info.iloc[0]['forename']} {driver_info.iloc[0]['surname']}"
                                    driver_laps = race_laps[race_laps['driverId'] == driver_id]
                                    
                                    # Convert lap times to strings with format "minute:second.millisecond"
                                    lap_times = []
                                    for _, lap in driver_laps.iterrows():
                                        millisec = lap['milliseconds']
                                        minutes = millisec // 60000
                                        seconds = (millisec % 60000) / 1000
                                        lap_times.append(f"{minutes}:{seconds:.3f}")
                                    
                                    race_info['lap_times'][driver_name] = lap_times
                
                races_info.append(race_info)
            
            logger.info(f"Extracted information for {len(races_info)} races")
        
        except Exception as e:
            logger.error(f"Error extracting race information: {e}")
    
    return races_info

def prepare_data_for_rag(race_info, output_file):
    """
    Prepare race information for the RAG model and save it to a file
    
    Args:
        race_info (list): List of race information dictionaries
        output_file (str): Path to save the processed data
        
    Returns:
        list: List of processed examples for the RAG model
    """
    logger.info(f"Preparing {len(race_info)} races for the RAG model")
    
    # Create examples for the RAG model
    examples = []
    
    for race in race_info:
        # Skip races without lap times
        if not race['lap_times']:
            continue
            
        # Create a detailed example for each driver
        for driver, lap_times in race['lap_times'].items():
            if not lap_times:
                continue
                
            example = {
                "race_id": race['race_id'],
                "race_name": race['race_name'],
                "circuit_name": race['circuit_name'],
                "location": race['circuit_location'],
                "date": race['date'],
                "year": race['year'],
                "driver": driver,
                "lap_times": lap_times[:10]  # Limit to first 10 laps
            }
            
            examples.append(example)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the processed data to a JSON file
    with open(output_file, 'w') as f:
        json.dump(examples, f, indent=2)
    
    logger.info(f"Saved {len(examples)} examples to {output_file}")
    
    # Also create text examples for the model
    text_examples = []
    for example in examples:
        text = f"Race: {example['race_name']} ({example['year']})\n"
        text += f"Circuit: {example['circuit_name']} - {example['location']}\n"
        text += f"Driver: {example['driver']}\n"
        text += f"Date: {example['date']}\n"
        text += "Lap Times: " + ", ".join(example['lap_times']) + "\n"
        text_examples.append(text)
    
    # Save text examples
    text_file = output_file.replace('.json', '_text.json')
    with open(text_file, 'w') as f:
        json.dump(text_examples, f, indent=2)
        
    logger.info(f"Saved {len(text_examples)} text examples to {text_file}")
    
    return text_examples

def main():
    """Main function for data processing"""
    parser = argparse.ArgumentParser(description="Process Formula 1 data for RAG model")
    parser.add_argument("--input", type=str, default="data/raw/archive", 
                        help="Directory containing CSV files")
    parser.add_argument("--output", type=str, default="data/processed/race_data.json",
                        help="Output file for processed data")
    
    args = parser.parse_args()
    
    # Load CSV data
    data_dict = load_csv_data(args.input)
    
    if not data_dict:
        logger.error("No data loaded, exiting.")
        return
    
    # Extract race information
    race_info = extract_race_information(data_dict)
    
    if not race_info:
        logger.error("No race information extracted, exiting.")
        return
    
    # Prepare data for the RAG model
    examples = prepare_data_for_rag(race_info, args.output)
    
    logger.info("Data processing complete")

if __name__ == "__main__":
    main()
