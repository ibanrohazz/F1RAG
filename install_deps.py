"""
Helper script to install required dependencies for F1RAG
"""
import sys
import subprocess
import importlib
import os
import platform

def check_python_version():
    """Check if Python version is compatible"""
    print(f"Python version: {sys.version}")
    if sys.version_info.major < 3 or (sys.version_info.major == 3 and sys.version_info.minor < 7):
        print("ERROR: Python 3.7 or newer is required")
        return False
    return True

def install_package(package, version=None):
    """Install a package with pip"""
    package_spec = f"{package}=={version}" if version else package
    
    print(f"Installing {package_spec}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to install {package_spec}, trying without version constraint...")
        try:
            if version:  # Only retry without version if version was specified
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                return True
            return False
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}")
            return False

def check_package(package):
    """Check if a package is installed"""
    try:
        importlib.import_module(package)
        return True
    except ImportError:
        return False

def create_directories():
    """Create required directories"""
    dirs = [
        "data/raw",
        "data/processed",
        "data/output",
        "data/passage_data",
        "models"
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def main():
    """Main function to install dependencies"""
    print("==================================================")
    print("F1RAG - Installing Dependencies")
    print("==================================================")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Upgrade pip
    print("Upgrading pip...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    except subprocess.CalledProcessError:
        print("WARNING: Failed to upgrade pip")
    
    # Define required packages with optional version constraints
    packages = [
        ("torch", "2.0.0"),
        ("transformers", "4.30.0"),
        ("datasets", None),
        ("scikit-learn", None),
        ("pandas", None),
        ("numpy", None),
        ("matplotlib", None),
        ("seaborn", None),
        ("tqdm", None)
    ]
    
    # Install packages
    failed_packages = []
    for package, version in packages:
        if not check_package(package):
            if not install_package(package, version):
                failed_packages.append(package)
        else:
            print(f"{package} is already installed")
    
    # Create directories
    create_directories()
    
    # Report results
    print("\n==================================================")
    if failed_packages:
        print(f"WARNING: Failed to install {len(failed_packages)} packages:")
        for package in failed_packages:
            print(f"  - {package}")
        print("\nPlease install these packages manually:")
        for package in failed_packages:
            print(f"  pip install {package}")
        print("\nAlternatively, try running the script again with admin/root privileges.")
    else:
        print("All dependencies installed successfully!")
    
    # Special note for transformers
    if "transformers" in failed_packages:
        print("\nNOTE: The 'transformers' package is critical for this project.")
        print("Try installing it manually with:")
        print("  pip install transformers")
    
    print("\nYou can now run the project with:")
    print("  python src/data_processing.py")
    print("  python src/rag_model.py")
    print("==================================================")

if __name__ == "__main__":
    main()
