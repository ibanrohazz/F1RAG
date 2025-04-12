@echo off
echo Creating new virtual environment for F1RAG...

:: Remove old environment if it exists
if exist env (
    echo Removing old environment...
    rmdir /s /q env
)

:: Create new environment with Python 3.9 if available
py -3.9 -m venv new_env 2>nul
if %errorlevel% neq 0 (
    echo Python 3.9 not available, trying with default Python...
    python -m venv new_env
)

:: Activate environment
echo Activating environment...
call new_env\Scripts\activate

:: Update pip and install packages
echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing dependencies...
pip install pandas numpy transformers torch matplotlib seaborn scikit-learn

:: Create directory structure for data
echo Creating directory structure...
mkdir data\raw 2>nul
mkdir data\processed 2>nul

echo.
echo Please select a data source:
echo 1. Generate sample F1 data
echo 2. Process real F1 archive data from data\raw\archive folder
echo 3. Skip data preparation for now
set /p data_choice=Enter your choice (1-3): 

if "%data_choice%"=="1" (
    echo Generating sample race data...
    python scripts\generate_sample_data.py
) else if "%data_choice%"=="2" (
    echo Processing F1 archive data...
    
    if not exist data\raw\archive (
        echo Error: data\raw\archive folder not found.
        echo Please ensure your F1 data archive is in the folder data\raw\archive.
        echo Archive structure should contain CSV files for races, lap times, drivers, etc.
    ) else (
        python scripts\process_f1_archive.py --archive data\raw\archive --output data\raw\race_data.csv
    )
)

echo.
echo Environment setup complete!
echo Run 'new_env\Scripts\activate' to activate the environment
echo.
echo Data sources for Formula 1 race data:
echo 1. Ergast API: http://ergast.com/mrd/ (Free F1 data API)
echo 2. Formula 1 Official API: https://www.formula1.com/en/f1-live.html (Requires subscription)
echo 3. F1 CSV datasets on Kaggle: https://www.kaggle.com/datasets?search=formula+1
echo 4. FastF1 Python package: pip install fastf1 (Python package for accessing F1 data)
echo.
echo To use the processed data, run: python src/data_processing.py
