# CVD-XAI: Explainable AI for Cardiovascular Disease Prediction

This repository presents an explainable AI approach for cardiovascular disease (CVD) prediction. The project combines deep learning (CNN, RNN) and tree-based algorithms (LightGBM, XGBoost, Random Forest) into an ensemble model, analyzed through SHAP and surrogate decision trees for interpretability.

---

## Overview

Cardiovascular diseases remain one of the most significant global health challenges. The goal of this project is to develop predictive models that achieve both **high performance** and **interpretability**.  
A series of models—ranging from baseline statistical learners to advanced neural networks—are trained, evaluated, and combined into a robust ensemble. The interpretability layer explains the ensemble’s decisions using explainable AI techniques.

---

## Project Structure

```
CVD-XAI/
│
├── Dataset/
│   ├── raw/
│   │   └── Heart Disease Health Indicators Dataset.csv
│   └── processed/
│       ├── feature_names.npy
│       ├── scaler.pkl
│       ├── X_train_scaled.npy
│       ├── X_test_scaled.npy
│       ├── y_train.npy
│       └── y_test.npy
│
├── Figures/
│   ├── Advanced Models/
│   ├── Baseline Models/
│   ├── Explanations/
│   └── Pre-processing/
│
├── Models/
│   ├── saved_models/
│   └── advanced_models/
│
└── Notebooks/
    ├── 01_Data_Preprocessing_and_Exploration_Notebook.ipynb
    ├── 02_Baseline_Models_Notebook.ipynb
    ├── 03_Advanced_Model_Development_Notebook.ipynb
    └── 04_Model_Interpretability_Analysis.ipynb
```

---

## Methodology

1. **Data Preprocessing** – The dataset is cleaned, scaled, and split into training and testing subsets.  
2. **Baseline Models** – Logistic Regression, Random Forest, LightGBM, and XGBoost are trained and compared.  
3. **Advanced Models** – Deep architectures (CNN and RNN) are developed and integrated with tree-based models to create an ensemble.  
4. **Explainability** – The ensemble model’s behavior is interpreted using:
   - **SHAP (SHapley Additive exPlanations)** for feature-level importance.
   - **Surrogate Decision Tree** for a global, human-understandable view of the model’s logic.

---

## Results and Insights

- The ensemble achieved improved predictive performance over individual models.  
- SHAP analysis identified key lifestyle and physiological features influencing CVD risk.  
- Surrogate tree visualization revealed interpretable global patterns behind predictions.

---

## Usage

1. Open the Jupyter notebooks in the **Notebooks/** directory to reproduce data preprocessing, model training, and analysis.  
2. Ensure dependencies are installed (see below).  
3. If any processed data files are missing, regenerate them by running the notebooks in sequence.

---

## Dependencies

- Python 3.10+
- NumPy  
- Pandas  
- Scikit-learn  
- TensorFlow / Keras  
- LightGBM  
- XGBoost  
- SHAP  
- Matplotlib / Seaborn  

To install all dependencies:
```bash
pip install -r requirements.txt
```
*(You can create a `requirements.txt` file containing the packages listed above.)*

---



---


---

## Contact

**Md Abrar Hasnat**  
BRAC University Graduate, Computer Science and Engineering  
Email: md.abrar.hasnat@g.bracu.ac.bd
