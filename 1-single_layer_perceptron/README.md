# Single Layer Perceptron (SLP) for Binary Classification

This repository contains the implementation of a Single Layer Perceptron from scratch to perform binary classification on the Banknote Authentication Dataset.

## Project Structure
- `slp.ipynb`: Jupyter notebook containing the dataset exploration, model implementation, training loop, and evaluation metrics.
- `data/`: Contains the `data_banknote_authentication.txt` dataset.
- `images/`: Directory where all generated plots (histograms, heatmaps, training errors, etc.) will be automatically saved upon execution.
- `requirements.txt`: Python package dependencies.

## Requirements

Install the necessary dependencies using pip:
```bash
pip install -r requirements.txt
```

## Execution Instructions

1. Ensure the dataset is present at `data/data_banknote_authentication.txt`.
2. Launch Jupyter Notebook in this directory:
   ```bash
   jupyter notebook slp.ipynb
   ```
3. Run all cells in the notebook sequentially (`Cell` > `Run All`).
4. All generated plots will automatically be saved to the `images/` directory in this folder.
