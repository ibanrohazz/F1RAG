import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_race_summaries(race_summaries):
    """
    Visualize race summaries using bar plots and line plots.
    
    Args:
        race_summaries (list): List of race summaries.
    """
    # Convert race summaries to DataFrame
    df = pd.DataFrame(race_summaries)
    
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
