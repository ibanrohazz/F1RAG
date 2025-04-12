# F1RAG

## Project Description

F1RAG is a project that uses Retrieval-Augmented Generation (RAG) to generate summaries of Formula 1 races. The project collects race data, including lap times, pit stops, overtakes, and incidents, and trains a RAG model to generate coherent and informative race summaries based on the collected data. The quality of the generated summaries is evaluated by comparing them with official race reports. The project also includes visualization tools and an interface for users to explore different races and their summaries.

## Environment Setup

To set up the environment for this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/githubnext/workspace-blank.git
   cd workspace-blank
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

To run the project, follow these steps:

1. Preprocess the race data:
   ```bash
   python src/data_processing.py
   ```

2. Train the RAG model:
   ```bash
   python src/rag_model.py
   ```

3. Generate race summaries:
   ```bash
   python src/rag_model.py --generate
   ```

4. Visualize the race summaries:
   ```bash
   python src/visualization.py
   ```

5. Explore the race summaries using the provided interface:
   Open the generated HTML file in your web browser.
