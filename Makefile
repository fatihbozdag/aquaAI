# Makefile for AquaAI

.PHONY: all install clean run-gui run-cli

# Default target
all: install

# Install dependencies
install:
	@echo "Installing dependencies from requirements.txt..."
	@pip install -r requirements.txt
	@echo "Installation complete."

# Run the GUI application
run-gui:
	@echo "Starting AquaAI GUI..."
	@python main.py

# Run the CLI application with example data
run-cli:
	@echo "Starting AquaAI CLI with example data..."
	@python main_cli.py --train-data ceyhan_normalize_veri.xlsx --test-data kuzey_ege_test_verisi.xlsx --models all

# Clean up the project
clean:
	@echo "Cleaning up project..."
	@rm -rf __pycache__
	@rm -rf */__pycache__
	@rm -rf results/*
	@echo "Cleanup complete."
