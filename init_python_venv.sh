#!/bin/bash

# # Define the Python version to use
# PYTHON_BIN=python3.11

# # Ensure the specified Python version is installed
# if ! command -v $PYTHON_BIN &> /dev/null; then
#     echo "$PYTHON_BIN is not installed. Please install it."
#     exit 1
# fi

# # Create a virtual environment if it doesn't exist
# if [ ! -d ".venv" ]; then
#     echo "Creating virtual environment with $PYTHON_BIN..."
#     $PYTHON_BIN -m venv .venv
# fi

# # Activate the virtual environment
# echo "Activating virtual environment..."
# source .venv/bin/activate

# # Upgrade pip and install dependencies
# echo "Installing dependencies..."
# $PYTHON_BIN -m pip install --upgrade pip
# $PYTHON_BIN -m pip install -r requirements.txt --break-system-packages

# # Install my_package
# echo "Installing my_package..."
# $PYTHON_BIN -m pip install -e ./python-scripts/


# Ensure Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python3."
    exit 1
fi

# Create a virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip and install dependencies
echo "Installing dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
# 以下、018へのロールバック用に残す（多分、そのままでもロールバック可能）
# pip install --upgrade pip
# pip install -r requirements.txt
# pip install -r requirements.txt --break-system-packages (仮想環境を使っていない場合にのみ必要)

# install my_package
echo "Installing my_package..."
python3 -m pip install -e ./python-scripts/
# 以下、018へのロールバック用に残す（多分、そのままでもロールバック可能）
# pip install -e ./python-scripts/

