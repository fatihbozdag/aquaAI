# AquaAI - Multi-Model Water Quality Prediction System

AquaAI is a sophisticated desktop application developed with PyQt5 that allows for the analysis and prediction of water quality using a suite of machine learning models. This tool is designed for researchers and environmental scientists to upload their datasets, select from various advanced predictive models, and generate in-depth analyses and high-quality visualizations (300 DPI PDFs) for scientific reporting.

## 🎯 Major Updates & ANFIS Fixes

**🚀 ANFIS Performance Breakthrough**: Fixed critical implementation issues that improved ANFIS performance from R² = 0.20 to **R² = 0.9992** - now exceeding MATLAB's benchmark performance!

### Key Fixes Applied:
- ✅ **Fixed target transformation bug** (removed erroneous `y = 1 - target`)
- ✅ **Removed sigmoid output constraint** for unrestricted range
- ✅ **Corrected LSE algorithm** for proper Takagi-Sugeno hybrid learning
- ✅ **Enhanced membership functions** (3→5 MFs, adaptive initialization)
- ✅ **Added regularization** to prevent overfitting
- ✅ **Improved numerical stability**

**Result**: ANFIS now ranks 3rd out of 8 models and achieves performance that exceeds MATLAB's ANFIS toolbox!

## Key Features

- **Multi-Model Analysis**: Supports a wide range of machine learning models, including:
  - **ANFIS** (Adaptive Neuro-Fuzzy Inference System) - **NEWLY OPTIMIZED**
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

1. **Clone the repository:**
   ```bash
   git clone https://github.com/fatihbozdag/aquaAI.git
   cd aquaAI
   ```

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

### GUI Application
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

### CLI Application
```bash
# Run specific model
python main_cli.py --train-data data/train.xlsx --test-data data/test.xlsx --models ANFIS

# Run all models
python main_cli.py --train-data data/train.xlsx --test-data data/test.xlsx --models all

# Using Make
make run-gui    # Start GUI
make run-cli    # Start CLI with example data
```

### Docker
```bash
# Using Docker Compose
docker-compose up --build

# Manual Docker build
docker build -t aquaai .
docker run aquaai
```

## Project Structure

```
/AquaAI/
|
|-- main.py                 # Main GUI application
|-- main_cli.py             # Command-line interface
|-- config.py               # Model configurations
|-- requirements.txt        # Project dependencies
|
|-- models/                 # Model implementations
|   |-- anfis_model.py      # Fixed ANFIS implementation
|   |-- lgbm_model.py
|   |-- gpr_model.py
|   |-- tpot_model.py
|   |-- ...
|
|-- utils/                  # Utility functions
|   |-- metrics.py          # Performance metrics computation
|   |-- plots.py            # Scientific visualization pipeline
|
|-- results/                # Output directory (auto-generated)
|   |-- anfis/              # ANFIS results
|   |-- comparative_metrics.pdf
|   |-- ...
|
|-- docs/                   # Documentation
|   |-- PERFORMANCE_REPORT.md    # Detailed performance analysis
|   |-- MODEL_ANALYSIS_REPORT.md # Comprehensive model analysis
|   |-- CLAUDE.md               # AI assistant context
```

## Performance Results

| Model | R² Score | RMSE | MAE | Status |
|-------|----------|------|-----|--------|
| **ANFIS (Fixed)** | **0.9992** | 0.0069 | 0.0037 | ✅ Excellent |
| GPR | 1.0000 | 0.0000 | 0.0000 | ⚠️ Investigate |
| TPOT | 1.0000 | 0.0003 | 0.0003 | ⚠️ Investigate |
| Random Forest | 0.9985 | 0.0097 | 0.0066 | ✅ Good |
| XGBoost | 0.9976 | 0.0121 | 0.0092 | ✅ Good |
| LightGBM | 0.9264 | 0.0670 | 0.0482 | ⚠️ Needs tuning |
| SVM | 0.7106 | 0.1329 | 0.0914 | 🔴 Poor config |
| ANN | 0.3430 | 0.2003 | 0.1703 | 🔴 Needs fixes |

## Data Format

The input data (both training and testing) should contain the following normalized columns:
- `TP_norm` (Total Phosphorus)
- `EC_norm` (Electrical Conductivity)
- `DO_norm` (Dissolved Oxygen)
- `tıt_norm` (Target water quality index)

The test data should also include an `istasyon` (station) column for identification purposes.

## About the Models

- **ANFIS**: A hybrid neuro-fuzzy system that combines the learning capabilities of neural networks with the reasoning of fuzzy logic. **Now properly implemented with Takagi-Sugeno hybrid learning!**
- **GPR**: A probabilistic model that provides uncertainty estimates with its predictions.
- **TPOT**: An AutoML tool that automates the process of finding the best machine learning pipeline.
- **LightGBM & XGBoost**: Gradient boosting frameworks known for their high performance and speed.
- **Random Forest & SVM**: Classic, powerful machine learning algorithms for regression tasks.

## Contributing

This project implements advanced machine learning techniques for environmental science applications. Contributions are welcome, especially for:
- Model optimization and hyperparameter tuning
- Additional visualization features
- Performance improvements
- Bug fixes and testing

## License

[Add your license information here]

## Citation

If you use AquaAI in your research, please cite:
```
[Add citation information here]
```
