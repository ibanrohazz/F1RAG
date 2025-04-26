def visualize_race_summaries(race_summaries):
    """
    Visualize race summaries using bar plots and line plots.
    Args:
        race_summaries (list): List of race summaries.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    df = pd.DataFrame(race_summaries)
    try:
        # Plot lap times
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='lap', y='lap_time', hue='driver')
        plt.title('Lap Times')
        plt.xlabel('Lap')
        plt.ylabel('Lap Time')
        plt.legend(title='Driver')
        plt.show()
        # Plot pit stops
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df[df['event'] == 'pit_stop'], x='lap', hue='driver')
        plt.title('Pit Stops')
        plt.xlabel('Lap')
        plt.ylabel('Count')
        plt.legend(title='Driver')
        plt.show()
        # Plot overtakes
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df[df['event'] == 'overtake'], x='lap', hue='driver')
        plt.title('Overtakes')
        plt.xlabel('Lap')
        plt.ylabel('Count')
        plt.legend(title='Driver')
        plt.show()
        # Plot incidents
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df[df['event'] == 'incident'], x='lap', hue='driver')
        plt.title('Incidents')
        plt.xlabel('Lap')
        plt.ylabel('Count')
        plt.legend(title='Driver')
        plt.show()
    except KeyboardInterrupt:
        print("Visualization interrupted by user. Exiting gracefully.")
        plt.close('all')
        return
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        plt.close('all')
        
def create_user_interface(race_summaries):
    """
    Create a user interface for exploring race summaries.
    
    Args:
        race_summaries (list): List of race summaries.
    """
    import tkinter as tk
    from tkinter import ttk

    def on_select(event):
        selected_race = race_listbox.get(race_listbox.curselection())
        summary_text.delete(1.0, tk.END)
        summary_text.insert(tk.END, race_summaries[selected_race])

    root = tk.Tk()
    root.title("Race Summaries")

    race_listbox = tk.Listbox(root)
    race_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    race_listbox.bind('<<ListboxSelect>>', on_select)

    summary_text = tk.Text(root)
    summary_text.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    for i, race in enumerate(race_summaries):
        race_listbox.insert(tk.END, f"Race {i+1}")

    root.mainloop()

def text_to_dataframe(summaries):
    """
    Convert text summaries to a structured DataFrame for visualization.
    
    Args:
        summaries (list): List of text summaries.
    
    Returns:
        DataFrame: Structured data for visualization.
    """
    import re
    import pandas as pd
    
    # Simple example conversion of text to structured data
    data = []
    
    for i, summary in enumerate(summaries):
        # Extract driver name (assuming it's mentioned)
        driver_match = re.search(r'Driver:?\s+([A-Za-z\s]+)', summary)
        driver = driver_match.group(1).strip() if driver_match else f"Driver {i+1}"
        
        # Try to extract lap times if present
        lap_times = re.findall(r'(\d+):(\d+\.\d+)', summary)
        
        if lap_times:
            for lap_idx, lap_time in enumerate(lap_times, 1):
                minutes, seconds = lap_time
                total_seconds = float(minutes) * 60 + float(seconds)
                data.append({
                    'driver': driver,
                    'lap': lap_idx,
                    'lap_time': total_seconds,
                    'event': 'lap_time'
                })
        else:
            # If no lap times, create a basic record
            data.append({
                'driver': driver,
                'lap': 1,
                'lap_time': 0,
                'event': 'summary'
            })
    
    return pd.DataFrame(data)

def load_summaries(file_path='data/output/race_summaries.json'):
    """
    Load race summaries from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file.
        
    Returns:
        list: List of race summaries.
    """
    import json
    import os
    
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found!")
        return create_sample_summaries()
        
    try:
        with open(file_path, 'r') as f:
            summaries = json.load(f)
            
        if not summaries:
            print("Warning: Empty summaries file")
            return create_sample_summaries()
            
        print(f"Loaded {len(summaries)} summaries from {file_path}")
        return summaries
        
    except Exception as e:
        print(f"Error loading summaries: {str(e)}")
        return create_sample_summaries()

def create_sample_summaries():
    """
    Create sample summaries for testing visualization.
    
    Returns:
        list: List of sample summaries.
    """
    print("Creating sample summaries for visualization...")
    
    return [
        "Driver: Hamilton\nRace: Monaco Grand Prix\nCircuit: Monaco\nLap Times: 1:12.345, 1:12.123, 1:11.987...",
        "Driver: Verstappen\nRace: Monaco Grand Prix\nCircuit: Monaco\nLap Times: 1:12.111, 1:11.876, 1:11.723...",
        "Driver: Leclerc\nRace: Monaco Grand Prix\nCircuit: Monaco\nLap Times: 1:12.267, 1:12.034, 1:11.954..."
    ]

def main():
    """
    Main function that runs when script is executed directly.
    """
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Visualize F1 race summaries")
    parser.add_argument("--file", type=str, default="data/output/race_summaries.json",
                       help="Path to race summaries JSON file")
    parser.add_argument("--ui", action="store_true", 
                       help="Show UI interface instead of plots")
    
    args = parser.parse_args()
    
    # Load the summaries
    summaries = load_summaries(args.file)
    
    if not summaries:
        print("No summaries available for visualization.")
        return
    
    try:
        if args.ui:
            # Launch UI
            create_user_interface(summaries)
        else:
            # Process text summaries into dataframe for visualization
            structured_data = text_to_dataframe(summaries)
            visualize_race_summaries(structured_data)
    except Exception as e:
        print(f"Error during visualization: {str(e)}")

# Run the main function when script is executed directly
if __name__ == "__main__":
    print("Starting F1 Race Visualization...")
    main()
