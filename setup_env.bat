@echo off
echo ==============================================
echo Formula 1 RAG System - Environment Setup
echo ==============================================

:: Remove old environment if it exists
if exist env (
    echo Removing old environment...
    rmdir /s /q env
)

:: Create new environment
echo Creating new virtual environment...
python -m venv env
if %errorlevel% neq 0 (
    echo Failed to create virtual environment.
    echo Please ensure Python is installed correctly.
    pause
    exit /b 1
)

:: Activate environment
echo Activating environment...
call env\Scripts\activate
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

:: Update pip and install packages
echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing dependencies...
echo This may take a few minutes...

:: Install packages one by one to ensure success
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cpu
if %errorlevel% neq 0 (
    echo Warning: Failed to install specific torch version. Trying without version constraint...
    pip install torch --index-url https://download.pytorch.org/whl/cpu
)

echo Installing transformers package...
pip install transformers==4.30.0
if %errorlevel% neq 0 (
    echo Failed with specific version, trying latest transformers...
    pip install transformers
)

:: Verify transformers was installed
python -c "import transformers; print('Transformers successfully installed:', transformers.__version__)"
if %errorlevel% neq 0 (
    echo ERROR: Failed to install transformers. Please try manual installation:
    echo pip install transformers
    pause
    exit /b 1
)

:: Install other required packages
echo Installing additional dependencies...
pip install datasets scikit-learn pandas numpy matplotlib seaborn tqdm
if %errorlevel% neq 0 (
    echo Warning: Some packages may have failed to install.
)

:: Create directory structure for data
echo Creating directory structure...
mkdir data\raw 2>nul
mkdir data\processed 2>nul
mkdir data\output 2>nul
mkdir data\passage_data 2>nul
mkdir models 2>nul

echo.
echo Please select a data source:
echo 1. Generate sample F1 data
echo 2. Process Formula 1 World Championship data from Kaggle (1950-2024)
echo 3. Skip data preparation for now
set /p data_choice=Enter your choice (1-3): 

if "%data_choice%"=="1" (
    echo Generating sample race data...
    python scripts\generate_sample_data.py
) else if "%data_choice%"=="2" (
    echo Processing F1 archive data...
    
    if not exist data\raw\archive (
        echo Error: data\raw\archive folder not found.
        echo Please download the Formula 1 dataset from Kaggle:
        echo https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020
        echo Extract the archive and place the CSV files in data\raw\archive folder.
        mkdir data\raw\archive 2>nul
    ) else (
        python src\data_processing.py --input data\raw\archive --output data\processed\race_data.json
    )
)

:: Create requirements file
pip freeze > requirements.txt

echo.
echo Environment setup complete!
echo.
echo How to use:
echo 1. Always activate the environment first:
echo    env\Scripts\activate
echo.
echo 2. Run the RAG model:
echo    python src/data_processing.py    (to process data)
echo    python src/rag_model.py          (to train the model)
echo    python src/rag_model.py --generate (to generate race summaries)
echo.
echo The Formula 1 dataset is from:
echo https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020
echo.

pause
