"""
Process Formula 1 archive data from the archive folder into the format
expected by the F1RAG system.
"""

import os
import pandas as pd
import glob
import argparse
import json
import numpy as np
from datetime import timedelta

def find_archive_files(archive_path):
    """Find all CSV files in the archive directory."""
    print(f"Searching for CSV files in {archive_path}...")
    csv_files = glob.glob(os.path.join(archive_path, "**", "*.csv"), recursive=True)
    
    # Group files by type based on filename
    file_categories = {
        'races': [f for f in csv_files if 'races' in os.path.basename(f).lower()],
        'lap_times': [f for f in csv_files if 'lap' in os.path.basename(f).lower()],
        'drivers': [f for f in csv_files if 'driver' in os.path.basename(f).lower()],
        'results': [f for f in csv_files if 'result' in os.path.basename(f).lower()],
        'pit_stops': [f for f in csv_files if 'pit' in os.path.basename(f).lower()],
        'status': [f for f in csv_files if 'status' in os.path.basename(f).lower()],
    }
    
    # Print summary of found files
    print(f"Found {len(csv_files)} CSV files:")
    for category, files in file_categories.items():
        print(f"  - {category}: {len(files)} files")
    
    return file_categories

def process_lap_times(lap_files, drivers_df):
    """Process lap time data from archive files."""
    print("Processing lap time data...")
    lap_data = []
    
    for file in lap_files:
        try:
            df = pd.read_csv(file)
            print(f"  - Processing {os.path.basename(file)}: {len(df)} records")
            
            # Check if required columns exist
            required_cols = ['raceId', 'driverId', 'lap', 'milliseconds', 'position']
            if all(col in df.columns for col in required_cols):
                lap_data.append(df)
            else:
                print(f"    Warning: Missing required columns in {file}. Skipping.")
        except Exception as e:
            print(f"    Error processing {file}: {e}")
    
    # Combine all lap data
    if lap_data:
        lap_df = pd.concat(lap_data, ignore_index=True)
        
        # Join with drivers to get driver names instead of IDs
        if 'driverId' in lap_df.columns and not drivers_df.empty:
            lap_df = lap_df.merge(drivers_df[['driverId', 'driverRef']], on='driverId', how='left')
            lap_df['driver'] = lap_df['driverRef']
        else:
            lap_df['driver'] = 'Driver_' + lap_df['driverId'].astype(str)
        
        # Convert milliseconds to timedelta
        lap_df['lap_time'] = lap_df['milliseconds'].apply(
            lambda x: str(timedelta(milliseconds=x))
        )
        
        return lap_df
    else:
        print("No valid lap time data found.")
        return pd.DataFrame()

def process_pit_stops(pit_files, drivers_df):
    """Process pit stop data from archive files."""
    print("Processing pit stop data...")
    pit_data = []
    
    for file in pit_files:
        try:
            df = pd.read_csv(file)
            print(f"  - Processing {os.path.basename(file)}: {len(df)} records")
            
            if 'lap' in df.columns and 'driverId' in df.columns:
                df['event'] = 'pit_stop'
                pit_data.append(df)
            else:
                print(f"    Warning: Missing required columns in {file}. Skipping.")
        except Exception as e:
            print(f"    Error processing {file}: {e}")
    
    if pit_data:
        pit_df = pd.concat(pit_data, ignore_index=True)
        
        # Join with drivers to get driver names instead of IDs
        if 'driverId' in pit_df.columns and not drivers_df.empty:
            pit_df = pit_df.merge(drivers_df[['driverId', 'driverRef']], on='driverId', how='left')
            pit_df['driver'] = pit_df['driverRef']
        else:
            pit_df['driver'] = 'Driver_' + pit_df['driverId'].astype(str)
            
        return pit_df
    else:
        print("No valid pit stop data found.")
        return pd.DataFrame()

def process_incidents(status_files, results_files, drivers_df):
    """Process incident data from status and results files."""
    print("Processing incident data...")
    
    # First try to get status data which contains retirements
    status_data = []
    for file in status_files:
        try:
            df = pd.read_csv(file)
            incident_statuses = df[df['status'].str.contains('Accident|Collision|Spun off|Damage|Crash|Failed', 
                                                            case=False, na=False)]
            if not incident_statuses.empty:
                status_data.append(incident_statuses)
        except Exception as e:
            print(f"    Error processing {file}: {e}")
    
    # Then process results files which can have status ID
    incidents_df = pd.DataFrame()
    if status_data:
        status_df = pd.concat(status_data, ignore_index=True)
        
        # Get results data which contains driver, race, and status info
        results_data = []
        for file in results_files:
            try:
                df = pd.read_csv(file)
                if 'statusId' in df.columns:
                    # Join with status to get only incidents
                    merged = df.merge(status_df, left_on='statusId', right_on='statusId', how='inner')
                    if not merged.empty:
                        merged['event'] = 'incident'
                        results_data.append(merged)
            except Exception as e:
                print(f"    Error processing {file}: {e}")
        
        if results_data:
            incidents_df = pd.concat(results_data, ignore_index=True)
            
            # Join with drivers to get driver names
            if 'driverId' in incidents_df.columns and not drivers_df.empty:
                incidents_df = incidents_df.merge(drivers_df[['driverId', 'driverRef']], 
                                               on='driverId', how='left')
                incidents_df['driver'] = incidents_df['driverRef']
            else:
                incidents_df['driver'] = 'Driver_' + incidents_df['driverId'].astype(str)
                
            # Set lap to last completed lap
            if 'laps' in incidents_df.columns:
                incidents_df['lap'] = incidents_df['laps']
            else:
                incidents_df['lap'] = 1  # Default value
    
    if not incidents_df.empty:
        print(f"Found {len(incidents_df)} incident records")
        return incidents_df
    else:
        print("No valid incident data found.")
        return pd.DataFrame()

def process_overtakes(lap_files, drivers_df):
    """Infer overtakes from position changes in lap data."""
    print("Inferring overtakes from position changes...")
    
    lap_data = []
    for file in lap_files:
        try:
            df = pd.read_csv(file)
            if 'position' in df.columns and 'driverId' in df.columns and 'lap' in df.columns:
                lap_data.append(df)
        except Exception as e:
            print(f"    Error processing {file}: {e}")
    
    overtakes_df = pd.DataFrame()
    if lap_data:
        lap_df = pd.concat(lap_data, ignore_index=True)
        
        # Sort by raceId, driverId, lap
        lap_df = lap_df.sort_values(['raceId', 'driverId', 'lap'])
        
        # Calculate position changes
        lap_df['prev_position'] = lap_df.groupby(['raceId', 'driverId'])['position'].shift(1)
        lap_df['position_improved'] = (lap_df['position'] < lap_df['prev_position'])
        
        # Filter for position improvements (overtakes)
        overtakes = lap_df[lap_df['position_improved']]
        if not overtakes.empty:
            overtakes['event'] = 'overtake'
            
            # Join with drivers to get driver names
            if 'driverId' in overtakes.columns and not drivers_df.empty:
                overtakes = overtakes.merge(drivers_df[['driverId', 'driverRef']], 
                                         on='driverId', how='left')
                overtakes['driver'] = overtakes['driverRef']
            else:
                overtakes['driver'] = 'Driver_' + overtakes['driverId'].astype(str)
                
            overtakes_df = overtakes[['raceId', 'driver', 'lap', 'event']]
    
    if not overtakes_df.empty:
        print(f"Inferred {len(overtakes_df)} overtake events")
        return overtakes_df
    else:
        print("No overtake data could be inferred.")
        return pd.DataFrame()

def load_driver_data(driver_files):
    """Load driver information from archive files."""
    print("Loading driver data...")
    drivers_df = pd.DataFrame()
    
    for file in driver_files:
        try:
            df = pd.read_csv(file)
            if 'driverId' in df.columns:
                if drivers_df.empty:
                    drivers_df = df
                else:
                    # Append new drivers not in our current set
                    new_drivers = df[~df['driverId'].isin(drivers_df['driverId'])]
                    if not new_drivers.empty:
                        drivers_df = pd.concat([drivers_df, new_drivers], ignore_index=True)
        except Exception as e:
            print(f"    Error loading {file}: {e}")
    
    if not drivers_df.empty:
        print(f"Loaded {len(drivers_df)} unique drivers")
        
        # Create driver reference if none exists
        if 'driverRef' not in drivers_df.columns:
            if 'forename' in drivers_df.columns and 'surname' in drivers_df.columns:
                drivers_df['driverRef'] = drivers_df['forename'].str[:1] + '_' + drivers_df['surname']
            else:
                drivers_df['driverRef'] = 'Driver_' + drivers_df['driverId'].astype(str)
    else:
        print("No driver data found.")
    
    return drivers_df

def combine_race_data(lap_df, pit_df, incident_df, overtake_df):
    """Combine all race data into a single DataFrame."""
    print("Combining race data...")
    
    # Prepare DataFrames with common columns
    dfs = []
    
    # Process lap times
    if not lap_df.empty:
        lap_df_subset = lap_df[['driver', 'lap', 'lap_time']].copy()
        lap_df_subset['event'] = 'normal'
        dfs.append(lap_df_subset)
    
    # Process pit stops
    if not pit_df.empty:
        pit_cols = ['driver', 'lap', 'event']
        if 'milliseconds' in pit_df.columns:
            pit_df['lap_time'] = pit_df['milliseconds'].apply(lambda x: str(timedelta(milliseconds=x)))
            pit_cols.append('lap_time')
        else:
            pit_df['lap_time'] = np.nan
            pit_cols.append('lap_time')
        dfs.append(pit_df[pit_cols])
    
    # Process incidents
    if not incident_df.empty:
        incident_cols = ['driver', 'lap', 'event']
        incident_df['lap_time'] = np.nan  # No lap time for incidents
        incident_cols.append('lap_time')
        dfs.append(incident_df[incident_cols])
    
    # Process overtakes
    if not overtake_df.empty:
        overtake_cols = ['driver', 'lap', 'event']
        overtake_df['lap_time'] = np.nan  # No specific lap time for overtakes
        overtake_cols.append('lap_time')
        dfs.append(overtake_df[overtake_cols])
    
    # Combine all data
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Sort by driver and lap
        combined_df = combined_df.sort_values(['driver', 'lap'])
        
        # Calculate positions (simple version - just for compatibility)
        # In a real implementation, we would calculate this more accurately
        drivers = combined_df['driver'].unique()
        position_map = {driver: i+1 for i, driver in enumerate(drivers)}
        combined_df['position'] = combined_df['driver'].map(position_map)
        
        print(f"Combined data has {len(combined_df)} records")
        return combined_df
    else:
        print("No data to combine!")
        return pd.DataFrame()

def export_data(combined_df, output_path, race_name="Formula1Race"):
    """Export the processed data to CSV file."""
    print(f"Exporting data to {output_path}...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    combined_df.to_csv(output_path, index=False)
    print(f"Data exported successfully with {len(combined_df)} records")
    
    # Print summary statistics
    print("\nData Summary:")
    print(f"Number of drivers: {combined_df['driver'].nunique()}")
    print(f"Number of laps: {combined_df['lap'].max()}")
    print("Event breakdown:")
    print(combined_df['event'].value_counts())
    
    return True

def process_archive_data(archive_path, output_path, race_id=None, season=None):
    """Process F1 archive data and prepare it for the F1RAG system."""
    print(f"Processing F1 archive data from {archive_path}")
    
    # Find all relevant files
    file_categories = find_archive_files(archive_path)
    
    # Load driver data first, as we'll need it to map IDs to names
    drivers_df = load_driver_data(file_categories.get('drivers', []))
    
    # Process lap times
    lap_df = process_lap_times(file_categories.get('lap_times', []), drivers_df)
    
    # Process pit stops
    pit_df = process_pit_stops(file_categories.get('pit_stops', []), drivers_df)
    
    # Process incidents (from status and results)
    incident_df = process_incidents(file_categories.get('status', []), 
                                  file_categories.get('results', []), 
                                  drivers_df)
    
    # Infer overtakes from position changes in lap data
    overtake_df = process_overtakes(file_categories.get('lap_times', []), drivers_df)
    
    # Combine all data
    combined_df = combine_race_data(lap_df, pit_df, incident_df, overtake_df)
    
    # Export the processed data
    if not combined_df.empty:
        success = export_data(combined_df, output_path)
        return success
    else:
        print("No data to export!")
        return False

def main():
    """Main function to process F1 archive data."""
    parser = argparse.ArgumentParser(description='Process F1 archive data for F1RAG.')
    parser.add_argument('--archive', type=str, default='archive',
                        help='Path to the F1 archive data folder')
    parser.add_argument('--output', type=str, default='data/raw/race_data.csv',
                        help='Path to save processed race data')
    parser.add_argument('--race', type=int, help='Specific race ID to process')
    parser.add_argument('--season', type=int, help='Specific season to process')
    
    args = parser.parse_args()
    
    success = process_archive_data(args.archive, args.output, args.race, args.season)
    
    if success:
        print("\nData processing completed successfully!")
        print(f"Processed data saved to {args.output}")
        print("You can now run the F1RAG system with this data:")
        print("  python src/data_processing.py")
    else:
        print("\nData processing failed!")

if __name__ == "__main__":
    main()
