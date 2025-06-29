
# config.py

# This file centralizes the configuration for models and other settings.

# Define available models and their properties
MODELS_CONFIG = {
    "ANFIS": {
        "name": "ANFIS",
        "class": "models.anfis_model.AnfisModel",
        "type": "custom",
        "params": {
            "num_mfs_per_input": 5 # Increased to 5 membership functions per input for better performance
        }
    },
    "ANN": {
        "name": "ANN (MLP)",
        "class": "sklearn.neural_network.MLPRegressor",
        "params": {
            "hidden_layer_sizes": (100, 50),
            "max_iter": 500,
            "random_state": 42
        },
        "type": "sklearn"
    },
    "LightGBM": {
        "name": "LightGBM",
        "class": "models.lgbm_model.LGBMModel",
        "type": "custom"
    },
    "GPR": {
        "name": "Gaussian Process Regression",
        "class": "models.gpr_model.GPRModel",
        "type": "custom"
    },
    "TPOT": {
        "name": "TPOT (AutoML)",
        "class": "models.tpot_model.TPOTModel",
        "type": "custom"
    },
    "RandomForest": {
        "name": "Random Forest",
        "class": "sklearn.ensemble.RandomForestRegressor",
        "params": {
            "n_estimators": 100,
            "random_state": 42
        },
        "type": "sklearn"
    },
    "XGBoost": {
        "name": "XGBoost",
        "class": "xgboost.XGBRegressor",
        "params": {
            "n_estimators": 100,
            "random_state": 42,
            "verbosity": 0
        },
        "type": "sklearn"
    },
    "SVM": {
        "name": "SVM",
        "class": "sklearn.svm.SVR",
        "params": {
            "kernel": "rbf"
        },
        "type": "sklearn"
    }
}

# Columns required for data loading
REQUIRED_COLUMNS = {
    "train": ['TP_norm', 'EC_norm', 'DO_norm', 'tıt_norm'],
    "test": ['TP_norm', 'EC_norm', 'DO_norm', 'tıt_norm', 'istasyon']
}

# Feature columns for training
FEATURE_COLUMNS = ['TP_norm', 'EC_norm', 'DO_norm', 'tıt_norm']

# Target column transformation
TARGET_COLUMN = 'tıt_norm'
