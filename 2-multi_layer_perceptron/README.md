# Deep Learning Laboratory: Experiment 2

This repository contains the source code and experimental results for Experiment 2 of the Deep Learning Laboratory (CS3807) course. It includes implementations for a Multi-Layer Perceptron (MLP) on the Fashion-MNIST dataset and an analysis of a single-layer perceptron on the XOR logic gate.

## Contents

1. **Multi-Layer Perceptron (MLP) for Multi-Class Classification** (`mlp.ipynb`)
   - Preprocesses the Fashion-MNIST image dataset (flattening, normalizing, and one-hot encoding).
   - Constructs and trains a baseline MLP using TensorFlow/Keras.
   - Evaluates the model using Accuracy, Precision, Recall, and F1-score.
   - Performs automated hyperparameter optimization using `RandomizedSearchCV` and `SciKeras`.
   - Generates confusion matrices and loss/accuracy plots.

2. **Perceptron for XOR Gate** (`xor.ipynb`)
   - Implements a single-layer perceptron learning algorithm from scratch in Python.
   - Evaluates the algorithm on the non-linearly separable XOR gate.
   - Plots decision boundaries for various weight updates.
   - Analyzes why the perceptron fails to converge for XOR.

3. **Report** (`report.pdf`)
   - The compiled report integrating the results, hyperparameters, plots, and analysis for both experiments.

4. **Images**
   - `images/`: Contains all plots generated from the Fashion-MNIST MLP training and hyperparameter search.
   - `images_xor/`: Contains visualizations of the decision boundary updates during the XOR perceptron training.

## Installation and Setup

1. **Clone the repository:**
   ```bash
   git clone <YOUR-REPO-LINK>
   cd 2-multi_layer_perceptron
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the notebooks:**
   ```bash
   jupyter notebook
   ```
   Open `mlp.ipynb` and `xor.ipynb` in your browser.

## Requirements
The project relies on standard data science and deep learning libraries, specifically tailored to use `scikeras` for hyperparameter tuning. See `requirements.txt` for exact versions.

- Python 3.8+
- TensorFlow / Keras
- SciKeras
- Scikit-Learn
- Numpy, Pandas, Matplotlib, Seaborn
