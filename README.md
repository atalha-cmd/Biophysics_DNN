
# Biophysics Machine Learning

Welcome to the **Biophysics Machine Learning** repository.....! 

This project presents a comprehensive pipeline for **feature generation** and **machine learning model training** designed to predict the biophysical properties of proteins. By integrating **electrostatic** and **topological** features within **Convolutional Neural Network (CNN)** architectures, the framework enables accurate prediction of electrostatic energies, including **Coulomb energy** and **solvation energy**.

---

## 🚀 Overview

The repository is organized into three major components:

1. **Electrostatic Feature Generation**  
   Tools to compute electrostatic features using a treecode-accelerated solver written in Fortran. These features capture electrostatic interactions that are crucial to protein properties.

2. **Topological Feature Generation**  
   Scripts to extract topological features (e.g., persistence diagrams, Betti numbers) that describe the **shape and connectivity** of biomolecular structures using topological data analysis (TDA).

3. **CNN Model Training**  
   Deep learning scripts for training **Convolutional Neural Networks** on electrostatic, topological, or combined features. Includes cross-validation for robust model evaluation.

---

## ⚙️ Prerequisites

Before using this repository, make sure your system is set up with the following:

- **Python Environment**:  
  Python **3.9+** (recommended)

- **Required Python Packages** (with tested versions):  
  - `numpy >= 1.23`  
  - `pandas >= 1.5`  
  - `matplotlib >= 3.6`  
  - `scikit-learn >= 1.3`  
  - `tensorflow >= 2.12` or `torch >= 2.0` (depending on your deep learning backend)  
  - `gudhi >= 3.7` (for persistent homology and topological features)

- **Fortran Compiler**:  
  `gfortran` (tested with GCC 11+) is required to run the numerical routines for electrostatics.

- **(Optional but Recommended)**:  
  Create and activate a **virtual environment** to manage dependencies:
  ```bash
  python -m venv env
  source env/bin/activate  # On Linux/Mac
  .\env\Scripts\activate   # On Windows


### 📂 Repository Structure

This repository includes several scripts designed for generating features and training machine learning models for biophysics research. Below is a summary of each script's purpose and how to use them.


## 🧑‍💻 Usage

Below are the key steps for generating features and training models:

### Step: 1. Generate Electrostatic Features

This script computes electrostatic features for biomolecules using a treecode-based Fortran solver.

```bash
python get_electrostatic_features.py 0 1 
```
Arguments:

`p` = interpolation order (e.g., 0)

`L` = tree depth (e.g., 1)

### Step: 2. Generate Topological Features 

This script computes topological features, which describes the shape and connectivity of the biomolecules.

```bash
python run_all.py  # Generate topological features
```
### Step: 3. Train CNN Models  

This script runs multiple CNN models using both electrostatic and topological features. It includes cross-validation for performance evaluation.

**Combined Features (Electrostatic + Topological):**
```bash
python CNNbothCV.py 0 1 5  # Run CNN with both features and cross-validation
```
``` 0 1 5 ``` : Parameters for interpolation order, tree depth, and number of cross-validation folds

**Electrostatic Features Only:**
```bash
python CNNelectro.py 0 1 5  # Run CNN with only electrostatic features and cross-validation
```
**Topological Features Only:**
```bash
python CNNtopologicalCV.py 5  # Run CNN with only topological features and cross-validation
```
## 📊 Output

- Feature files: Saved as CSV or NumPy arrays for reproducibility.

- Model checkpoints: Saved CNN models for reuse and evaluation.

- Performance metrics: Includes MSE, R², Pearson correlation, and cross-validation scores.

- Plots: Training/validation loss curves and scatter plots of predictions vs. true values.

## 📖 References

If you use this repository in your work, please consider citing:

- Topological feature extraction: GUDHI Library (https://gudhi.inria.fr/python/latest/)

- Electrostatics solver (Treecode implementation): Related literature in computational biophysics and electrostatics

- CNN models: Built using TensorFlow
