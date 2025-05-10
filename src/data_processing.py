"""
Data processing for Formula 1 RAG retrieval system.
Provides functions to load CSV files and build winning fact statements.
"""
import pandas as pd

def load_f1_data(data_dir='data/raw/archive'):
    """
    Load core Formula 1 datasets from CSV files.

    Args:
        data_dir (str): Path to the directory containing races.csv, drivers.csv, results.csv, constructors.csv
    Returns:
        tuple: (races, drivers, results, constructors) DataFrames
    """
    races = pd.read_csv(f"{data_dir}/races.csv")
    drivers = pd.read_csv(f"{data_dir}/drivers.csv")
    results = pd.read_csv(f"{data_dir}/results.csv")
    constructors = pd.read_csv(f"{data_dir}/constructors.csv")
    return races, drivers, results, constructors

def build_f1_facts(races, drivers, results, constructors):
    """
    Merge datasets and extract winning facts for each Grand Prix.

    Args:
        races (DataFrame): Races data
        drivers (DataFrame): Drivers data
        results (DataFrame): Results data
        constructors (DataFrame): Constructors data
    Returns:
        list: List of fact strings
    """
    # Merge to create combined race data
    df = results.merge(races, on='raceId')\
                .merge(drivers, on='driverId')\
                .merge(constructors, on='constructorId')
    # Filter for winners
    winners = df[df['positionOrder'] == 1].copy()
    # Build fact statements
    winners['fact'] = winners.apply(
        lambda row: f"In {row['year']}, {row['forename']} {row['surname']} won the {row['name_x']} driving for {row['name_y']}.",
        axis=1
    )
    return winners['fact'].tolist()
