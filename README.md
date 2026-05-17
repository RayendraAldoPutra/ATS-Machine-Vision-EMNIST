# ATS-Machine-Vision-EMNIST
# Machine Vision Midterm Project: EMNIST Letters Classification

**Author:** Rayendra Aldo Putra  
**Student ID (NIM):** 4222301064  
**Study Program:** Robotics Engineering  

## Project Overview
This repository contains the source code for the Midterm Assessment (Asesmen Tengah Semester) of the Machine Vision course. The project focuses on building a complete machine learning pipeline to classify handwritten characters (A-Z) using traditional computer vision techniques. 

The pipeline utilizes **Histogram of Oriented Gradients (HOG)** for feature extraction and a **Support Vector Machine (SVM)** optimized via Grid Search for classification.

## Dataset
* **Source:** [EMNIST (Extended MNIST) Dataset](https://www.kaggle.com/datasets/crawford/emnist/data)
* **Specific File:** `emnist-letters-train.csv`
* **Data Preparation:** The dataset is balanced by randomly sampling exactly 100 images for each of the 26 alphabet classes, resulting in a total of 2,600 samples.

## Machine Learning Pipeline
1. **Data Preprocessing:** The 1D pixel arrays (784 pixels) are reshaped into 28x28 2D matrices. A transpose operation (`.T`) is applied to correct the default rotated and flipped orientation of the EMNIST dataset.
2. **Feature Extraction (HOG):** The default parameters were modified to extract optimal structural features. 
   * `orientations`: 9
   * `pixels_per_cell`: (7, 7)
   * `cells_per_block`: (2, 2)
3. **Classification (SVM + Grid Search):** The dataset is split into 80% training and 20% testing sets using stratified sampling. An SVM classifier is trained using GridSearchCV (5-fold Cross-Validation) to find the best hyperparameters.
   * **Best Parameters Found:** `Kernel: RBF`, `C: 10`, `Gamma: scale`.

## Evaluation & Results
The model was evaluated on the unseen 20% test data (520 samples). The final performance metrics are as follows:
* **Accuracy:** 84.23%
* **Precision:** ~84%
* **Recall:** ~84%
* **F1-Score:** ~84%

A detailed Classification Report and a Confusion Matrix heatmap are generated within the notebook to visualize the model's performance across all 26 classes.

## How to Run
1. Clone this repository.
2. Download the `emnist-letters-train.csv` file from Kaggle and place it in the same directory as the notebook.
3. Install the required dependencies: `pip install pandas numpy scikit-learn scikit-image matplotlib seaborn`.
4. Run all cells in the Jupyter Notebook (`.ipynb`).
