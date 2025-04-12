"""
Generate sample Formula 1 race data for the F1RAG project.
This script creates a CSV file with synthetic race data including drivers,
lap times, and events such as pit stops, overtakes, and incidents.
"""

import os
import pandas as pd
import numpy as np
import random
from datetime import timedelta

def generate_sample_race_data(output_file, num_drivers=20, num_laps=60):
    """Generate sample race data and save to a CSV file."""
    
    # Create list of fictional drivers
    drivers = [f"Driver_{i+1}" for i in range(num_drivers)]
    
    # Generate data
    data = []
    
    for lap in range(1, num_laps + 1):
        for driver in drivers:
            # Basic lap data
            base_time = 90 + np.random.normal(0, 1)  # Average lap time of 90 seconds with some variation
            
            # Add some variance by driver (some drivers consistently faster/slower)
            driver_skill = np.random.normal(0, 2)  # Driver-specific skill factor
            
            # Add some variance by lap (tire degradation, fuel load changes)
            lap_factor = 0.02 * lap  # Slight increase in lap times over the race
            
            # Calculate lap time in seconds
            lap_time_seconds = base_time + driver_skill + lap_factor
            
            # Convert to timedelta format (MM:SS.sss)
            lap_time = str(timedelta(seconds=lap_time_seconds))
            
            # Basic lap record
            record = {
                'driver': driver,
                'lap': lap,
                'lap_time': lap_time,
                'position': 0,  # Will be calculated later
                'event': 'normal'
            }
            data.append(record)
            
            # Generate special events
            # Pit stops: typically every ~20 laps per driver
            if lap % 20 == random.randint(0, 5) and lap > 5:
                pit_record = record.copy()
                pit_record['event'] = 'pit_stop'
                pit_record['lap_time'] = str(timedelta(seconds=lap_time_seconds + 25))  # Slower lap time due to pit
                data.append(pit_record)
            
            # Random overtakes
            if random.random() < 0.03:  # 3% chance of overtake per lap per driver
                overtake_record = record.copy()
                overtake_record['event'] = 'overtake'
                data.append(overtake_record)
            
            # Random incidents (crashes, spins, etc)
            if random.random() < 0.01:  # 1% chance of incident per lap per driver
                incident_record = record.copy()
                incident_record['event'] = 'incident'
                incident_record['lap_time'] = str(timedelta(seconds=lap_time_seconds + 10))  # Slower lap time due to incident
                data.append(incident_record)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Sample race data saved to {output_file}")
    return df

if __name__ == "__main__":
    output_file = "data/raw/race_data.csv"
    generate_sample_race_data(output_file)
    
    # Print a sample of the generated data
    df = pd.read_csv(output_file)
    print("\nSample of generated data:")
    print(df.head(10))
    
    # Print some statistics
    print("\nData statistics:")
    print(f"Total records: {len(df)}")
    print(f"Number of drivers: {df['driver'].nunique()}")
    print(f"Number of laps: {df['lap'].max()}")
    print(f"Events breakdown:")
    print(df['event'].value_counts())
