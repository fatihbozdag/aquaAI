# AquaAI - Multi-Model Water Quality Prediction System

AquaAI is a sophisticated desktop application developed with PyQt5 that allows for the analysis and prediction of water quality using a suite of machine learning models. This tool is designed for researchers and environmental scientists to upload their datasets, select from various advanced predictive models, and generate in-depth analyses and high-quality visualizations (300 DPI PDFs) for scientific reporting.

## Key Features

- **Multi-Model Analysis**: Supports a wide range of machine learning models, including:
  - **ANFIS** (Adaptive Neuro-Fuzzy Inference System)
  - **ANN** (Artificial Neural Network - MLP)
  - **LightGBM**
  - **GPR** (Gaussian Process Regression)
  - **TPOT** (Tree-based Pipeline Optimization Tool - AutoML)
  - **Random Forest**
  - **XGBoost**
  - **SVM** (Support Vector Machine)
- **User-Friendly Interface**: An intuitive GUI built with PyQt5 for easy data loading, model selection, and analysis execution.
- **Comprehensive Visualizations**: Automatically generates a variety of scientific-grade plots for each model, such as:
  - Scatter plots of predicted vs. actual values
  - Residual analysis plots
  - SHAP value visualizations for model interpretability
  - Confusion matrices
  - GPR confidence intervals
  - ANFIS fuzzy surfaces
- **Comparative Analysis**: Generates a comparative metrics plot to evaluate the performance of all selected models against each other.
- **Data Input**: Supports both Excel (`.xlsx`, `.xls`) and CSV (`.csv`) file formats for training and testing data.
- **Organized Output**: All results, including performance metrics and visualizations, are systematically saved in a `results/` directory, organized by model.

## Installation

To run AquaAI, you need Python 3 and the dependencies listed in `requirements.txt`.

1. **Clone the repository or download the source code.**

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

1. **Activate the virtual environment** (if you created one).

2. **Run the main application:**
   ```bash
   python main.py
   ```

3. **Using the Application:**
   - Click **"Eğitim Verisi Yükle"** to load your training dataset.
   - Click **"Test Verisi Yükle"** to load your testing dataset.
   - **Select the models** you wish to analyze from the checkboxes.
   - Click **"Seçili Modelleri Analiz Et"** to start the analysis.
   - The progress bar will indicate the status, and upon completion, all results will be available in the `results/` folder.

## Project Structure

```
/ANFIS_CURSOR/
|
|-- main.py                 # Main application script (GUI and workflow)
|-- requirements.txt        # Project dependencies
|
|-- models/                 # Directory for model implementations
|   |-- anfis_model.py
|   |-- lgbm_model.py
|   |-- gpr_model.py
|   |-- tpot_model.py
|   |-- ...
|
|-- utils/                  # Utility functions
|   |-- metrics.py          # Functions for computing performance metrics
|   |-- plots.py            # Functions for generating visualizations
|
|-- results/                # Directory for output files
|   |-- anfis/
|   |-- ann/
|   |-- lgbm/
|   |-- ...
|
|-- *.xlsx                  # Example data files
```

## Data Format

The input data (both training and testing) should contain the following normalized columns:
- `TP_norm` (Total Phosphorus)
- `EC_norm` (Electrical Conductivity)
- `DO_norm` (Dissolved Oxygen)
- `tıt_norm` (A target variable, likely related to a water quality index)

The test data should also include an `istasyon` (station) column for identification purposes.

## About the Models

- **ANFIS**: A hybrid neuro-fuzzy system that combines the learning capabilities of neural networks with the reasoning of fuzzy logic.
- **GPR**: A probabilistic model that provides uncertainty estimates with its predictions.
- **TPOT**: An AutoML tool that automates the process of finding the best machine learning pipeline.
- **LightGBM & XGBoost**: Gradient boosting frameworks known for their high performance and speed.
- **Random Forest & SVM**: Classic, powerful machine learning algorithms for regression tasks.
