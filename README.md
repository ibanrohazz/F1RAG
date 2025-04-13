# F1RAG - Formula 1 Retrieval Augmented Generation System

F1RAG is a retrieval-augmented generation system focused on Formula 1 racing data. The system leverages historical Formula 1 data to provide insights, answer questions, and generate content related to Formula 1 racing.

## Data Source

The system uses Formula 1 World Championship data from 1950-2020, sourced from:
- **Dataset**: [Formula 1 World Championship (1950-2020)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
- **Platform**: Kaggle
- **Credit**: Rohan Rao (Dataset Creator)

## Setup and Requirements

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/F1RAG.git
   cd F1RAG
   ```

2. Quick setup (recommended):
   ```bash
   # On Windows
   setup_env.bat
   
   # On any platform with Python
   python install_deps.py
   ```

3. Manual setup:
   ```bash
   # Create and activate a virtual environment
   python -m venv env
   
   # On Windows
   env\Scripts\activate
   
   # On macOS/Linux
   source env/bin/activate
   
   # Install required packages
   pip install torch transformers datasets scikit-learn pandas matplotlib seaborn
   ```

4. Download the dataset:
   - Download the [Formula 1 dataset from Kaggle](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
   - Extract the CSV files into the `data/raw/archive` directory

## Running the Project

Always activate the virtual environment first:
```bash
# On Windows
env\Scripts\activate

# On macOS/Linux
source env/bin/activate
```

Then follow these steps:

1. Process the race data:
   ```bash
   python src/data_processing.py
   ```

2. Train the model:
   ```bash
   python src/rag_model.py
   ```

3. Generate race summaries:
   ```bash
   python src/rag_model.py --generate
   ```

4. Visualize the race summaries (optional):
   ```bash
   python src/visualization.py
   ```

## Troubleshooting

If you encounter a `ModuleNotFoundError`, ensure you've:
1. Activated the virtual environment
2. Installed all dependencies 
3. Used the correct Python interpreter

To verify your environment, run:
```bash
python -c "import transformers; print('Transformers version:', transformers.__version__)"
```

If that fails, try reinstalling the transformers package:
```bash
pip install --upgrade transformers
```

## License

This project uses data under the terms provided by Kaggle and the dataset creator. 
The F1 data is provided for educational and research purposes.

## Acknowledgments

- Thanks to Rohan Rao for compiling and sharing the comprehensive Formula 1 dataset on Kaggle.
- Formula 1 (F1) for the exciting sport that generated this data.
