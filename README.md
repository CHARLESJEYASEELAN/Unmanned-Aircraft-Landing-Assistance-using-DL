# Aircraft Landing Dataset Simulation & Classification

This notebook demonstrates the process of simulating an aircraft landing dataset, preprocessing it, and applying both machine learning and deep learning classifiers to predict the PAPI (Precision Approach Path Indicator) Status. The workflow includes data generation, preprocessing, model training, evaluation, and visualization.

---

## Table of Contents

1. [Dataset Simulation](#dataset-simulation)
2. [Preprocessing](#preprocessing)
3. [Train-Test Split](#train-test-split)
4. [Machine Learning Classifiers](#machine-learning-classifiers)
    - Gaussian Naive Bayes
    - K-Nearest Neighbors (KNN)
    - Support Vector Machine (SVM)
5. [Deep Learning Classifiers](#deep-learning-classifiers)
    - Simple ANN
    - Parallel MLP
6. [Model Evaluation & Visualization](#model-evaluation--visualization)
7. [Saving Models & Data](#saving-models--data)
8. [Requirements](#requirements)
9. [Usage](#usage)
10. [Files Generated](#files-generated)

---

## Dataset Simulation

- **Features simulated:**
    - MEHT (Minimum Eye Height over Threshold) in feet
    - C Marker (feet)
    - Decision Altitude (feet)
    - Vertical Speed at Threshold (feet/min)
    - Wind Speed (knots)
    - Wind Direction (degrees)
    - Runway Condition (Dry, Wet, Snowy, Icy)
    - PAPI Status (High, Low, Mixed) — derived from MEHT and Vertical Speed

- **Process:**
    - Random values are generated for each feature using NumPy.
    - PAPI Status is determined by a custom function based on MEHT and Vertical Speed.
    - Data is shuffled and saved as `aircraft_landing_dataset.csv`.

---

## Preprocessing

- **Categorical Encoding:**  
  Runway Condition and PAPI Status are label-encoded using `LabelEncoder`.

- **Numerical Standardization:**  
  All numerical features are standardized using `StandardScaler`.

---

## Train-Test Split

- The dataset is split into training and testing sets (70% train, 30% test) using `train_test_split`.
- Features (`X`) and target (`y`) are separated.

---

## Machine Learning Classifiers

### 1. Gaussian Naive Bayes
- Trained on the preprocessed data.
- Evaluation metrics: Accuracy, Precision, Recall, F1 Score, Classification Report, Confusion Matrix (visualized with Seaborn).

### 2. K-Nearest Neighbors (KNN)
- Trained and evaluated similarly to Naive Bayes.
- Confusion matrix and metrics are visualized.

### 3. Support Vector Machine (SVM)
- Trained and evaluated as above.
- Results visualized.

- **Comparison:**  
  All models' metrics are plotted in a clustered bar chart for easy comparison.

---

## Deep Learning Classifiers

### 1. Simple ANN
- Built using TensorFlow/Keras `Sequential` API.
- Architecture: 4 hidden layers (128, 64, 32, 16 units), output layer (3 units, softmax).
- Trained for 75 epochs.
- Training/validation accuracy and loss are plotted.
- Evaluation metrics and confusion matrix are displayed.

### 2. Parallel MLP
- Custom model with three parallel MLP branches, concatenated before the output.
- Trained and evaluated similarly to the Simple ANN.
- Training/validation metrics are plotted.
- Confusion matrix and metrics are displayed.

---

## Model Evaluation & Visualization

- Accuracy, Precision, Recall, F1 Score, and Confusion Matrix are calculated for all models.
- Results are visualized using Matplotlib and Seaborn.

---

## Saving Models & Data

- Machine learning models are saved as `.pkl` files using `joblib`.
- Deep learning models are saved as `.h5` files using Keras.
- Train and test datasets are saved as `.npy` files.

---

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- tensorflow (for deep learning)
- joblib

Install dependencies:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib tensorflow joblib
```

---

## Usage

1. Run the notebook step by step to simulate the dataset, preprocess, train models, and visualize results.
2. Generated files (`.csv`, `.pkl`, `.h5`, `.npy`) will be saved in the working directory.
3. Use the saved models for inference or further analysis.

---

## Files Generated

- `aircraft_landing_dataset.csv` — Simulated dataset
- `gaussian_naive_bayes_model.pkl` — Saved Naive Bayes model
- `KNN.pkl` — Saved KNN model
- `SVC.pkl` — Saved SVM model
- `keras_model.h5` — Saved Simple ANN model
- `simple_ann.h5` — Saved Simple ANN model
- `parallel_mlp.h5` — Saved Parallel MLP model
- `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy` — Train/test splits

---

## Notes

- The notebook is modular and can be extended with more classifiers or feature engineering.
- All visualizations are generated inline for easy interpretation.
- The workflow is suitable for educational purposes and can be adapted for real-world datasets.

---
