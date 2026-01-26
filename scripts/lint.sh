#!/bin/bash

export PROJECT_DIR="$(dirname "$PWD")"
cd $PROJECT_DIR

# Sort imports (using the black profile to prevent conflicts)
echo ">>> Running isort..."
isort .

# Format code
echo ">>> Running black..."
black . 

# # Lint code (pointing to your source directory)
# echo ">>> Running pylint..."
# pylint utils/ main.py