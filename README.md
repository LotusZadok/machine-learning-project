# Machine Learning Project - Artificial Intelligence Course

Welcome to the **Machine Learning Project** repository for the **Artificial Intelligence** course. This project focuses on customer churn prediction using a structured machine learning pipeline, including data preprocessing, feature selection, model training, and hyperparameter tuning.

---

## 🛠️ Environment Setup

1. **Create and Activate the Virtual Environment**

```bash
python -m venv venv
.env\Scripts\activate
```

2. **Upgrade Core Packages**

```bash
pip install --upgrade pip setuptools wheel
```

3. **Install Project Dependencies**

```bash
pip install -r requirements.txt
```

4. **Set Up Jupyter Kernel (Optional)**

```bash
python -m ipykernel install --user --name=venv --display-name "ML Project"
```

5. **Start Jupyter Notebook (Optional)**

```bash
jupyter notebook
```

---

## 📁 Project Structure

```
machine-learning-project/
├── data/
│   ├── raw/                  # Raw, unprocessed data
│   └── processed/            # Preprocessed, cleaned data
│
├── docs/                     # Documentation and reports
│
├── notebooks/                # Jupyter notebooks for data exploration
│
├── reports/                  # Final project reports
│
├── src/                      # Machine learning scripts
│   ├── data/                 # Data preparation scripts
│   ├── eval/                 # Model evaluation scripts
│   ├── iterations/           # Iterative model improvements
│   ├── models/               # Model training scripts
│   ├── preprocessing/        # Data preprocessing scripts
│   ├── tuning/               # Hyperparameter tuning scripts
│   └── utils/                # Utility functions
│
├── models/                   # Saved model files (gitignored for convenience)
│
├── .gitignore                # Git ignore file
├── README.md                 # Project overview and setup guide
└── requirements.txt          # Project dependencies
```

---

## 🚀 Running Scripts

All scripts in the `src/` directory can be executed from the command line using the following format:

```bash
python src/<folder>/<script_name>.py
```

Examples:

```bash
python src/data/data_preparation.py
python src/models/xgboost.py
python src/tuning/xgboost_tuning.py
```

---

## 📂 Model Storage

- The `models/` folder is included in the `.gitignore` file for convenience, ensuring that large model files do not clutter the repository.
