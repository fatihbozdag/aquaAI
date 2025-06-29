import pandas as pd
import matplotlib.pyplot as plt

# Verini buraya kopyala
data = {
    "TIT": [
        2.36, 2.64, 1.55, 2.74, 2.59, 2.86, 2.36, 2.73, 3.18, 2.55,
        1.84, 2.99, 1.99, 1.38, 1.95, 1.95, 2.34, 1.89, 1.47, 2.59,
        2.90, 2.56, 2.47, 1.89, 2.08, 1.75, 1.98, 2.00, 3.20, 2.10,
        2.70, 2.62, 2.09, 1.91, 2.50, 2.28, 2.08, 1.98, 2.39, 2.14,
        1.57, 2.09, 1.90, 2.10, 1.76, 1.54, 2.81
    ],
    "ANFIS": [
        0.3137, 0.1996, 0.6371, 0.2556, 0.2304, 0.179, 0.4428, 0.2846, 0.071, 0.115,
        0.5676, 0.1138, 0.4896, 0.648, 0.6416, 0.595, 0.4033, 0.6038, 0.7717, 0.4043,
        0.2576, 0.5105, 0.4845, 0.4228, 0.537, 0.6901, 0.4062, 0.5163, 0.2972, 0.6137,
        0.4073, 0.3022, 0.5545, 0.1502, 0.263, 0.2138, 0.4773, 0.6162, 0.5017, 0.4691,
        0.6075, 0.5435, 0.6677, 0.518, 0.6482, 0.7572, 0.3176
    ]
}

df = pd.DataFrame(data)

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df['TIT'], df['ANFIS'], color='blue', alpha=0.7)
plt.title("TIT vs ANFIS Su Kalitesi Skoru")
plt.xlabel("TIT Skoru (↓ daha temiz su)")
plt.ylabel("ANFIS Skoru (↑ daha temiz su)")
plt.grid(True)
plt.tight_layout()
plt.show()
