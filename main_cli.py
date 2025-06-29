import argparse
import os
import pandas as pd
import numpy as np
import importlib
from config import MODELS_CONFIG, FEATURE_COLUMNS, TARGET_COLUMN
from utils.metrics import compute_metrics
from utils.plots import create_model_visualizations, plot_comparative_metrics

def load_data(file_path):
    """Loads data from an Excel or CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    return pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

def get_model_instance(model_name):
    """Dynamically imports and returns an instance of the specified model."""
    if model_name not in MODELS_CONFIG:
        raise ValueError(f"Unknown model: {model_name}")
        
    config = MODELS_CONFIG[model_name]
    module_path, class_name = config["class"].rsplit('.', 1)
    
    try:
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        
        if config["type"] == "sklearn":
            return model_class(**config.get("params", {}))
        else: # Custom models
            return model_class()
            
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not load model: {model_name}. Error: {e}")

def main(args):
    """Main function to run the CLI application."""
    print("Starting AquaAI CLI Analysis...")

    try:
        print(f"Loading training data from: {args.train_data}")
        training_data = load_data(args.train_data)
        print(f"Loading test data from: {args.test_data}")
        test_data = load_data(args.test_data)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    try:
        X_train = training_data[FEATURE_COLUMNS].values
        y_train = training_data[TARGET_COLUMN].values
        X_test = test_data[FEATURE_COLUMNS].values
        y_test = test_data[TARGET_COLUMN].values
        print("Data preparation complete.")
    except KeyError as e:
        print(f"Data preparation error: Missing required column - {e}")
        return

    selected_models = list(MODELS_CONFIG.keys()) if "all" in args.models else args.models
    print(f"Selected models for analysis: {', '.join(selected_models)}")

    all_metrics = {}
    
    for model_key in selected_models:
        print(f"\n{'='*20}\nAnalyzing {MODELS_CONFIG[model_key]['name']}...\n{'='*20}")
        try:
            model_config = MODELS_CONFIG[model_key]
            model = get_model_instance(model_key)
            
            if model_config["type"] == "custom":
                model.train(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics = model.evaluate(y_test, y_pred)
            else: # sklearn
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics = compute_metrics(y_test, y_pred)

            all_metrics[model_config['name']] = metrics
            
            output_dir = f"results/{model_key.lower()}"
            os.makedirs(output_dir, exist_ok=True)
            with open(f"{output_dir}/metrics.txt", "w", encoding='utf-8') as f:
                f.write(f"{model_config['name']} Model Performance Metrics\n")
                f.write("=" * 40 + "\n")
                for key, value in metrics.items():
                    f.write(f"{key}: {value:.4f}\n")
            print(f"Metrics saved to {output_dir}/metrics.txt")

            print("Creating visualizations...")
            create_model_visualizations(model_key, model, X_train, y_train, X_test, y_test, y_pred, output_dir)
            print(f"Visualizations saved in {output_dir}")

        except Exception as e:
            print(f"!!! Failed to analyze {model_key}: {e}")

    if len(all_metrics) > 1:
        print("\nGenerating comparative metrics plot...")
        try:
            plot_comparative_metrics(all_metrics, "results/comparative_metrics.pdf")
            print("Comparative metrics plot saved to results/comparative_metrics.pdf")
        except Exception as e:
            print(f"!!! Could not generate comparative plot: {e}")
            
    print("\nAquaAI CLI Analysis Finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AquaAI - Command-Line Water Quality Prediction")
    
    parser.add_argument('--train-data', type=str, required=True, help="Path to the training data file.")
    parser.add_argument('--test-data', type=str, required=True, help="Path to the test data file.")
    parser.add_argument('--models', nargs='+', required=True, help=f"List of models to analyze. Choices: {list(MODELS_CONFIG.keys()) + ['all']}")
    
    args = parser.parse_args()
    main(args)
