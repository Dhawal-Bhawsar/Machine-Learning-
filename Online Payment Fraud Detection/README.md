# Online Payment Fraud Detection

A machine learning project that analyzes online financial transactions and compares classification models for identifying potentially fraudulent payments.

The notebook explores transaction patterns, prepares features, and trains three classifiers: Logistic Regression, XGBoost, and Random Forest. It evaluates model predictions using ROC-AUC and visualizes a confusion matrix for Logistic Regression.

## Project Objective

Fraudulent transactions are uncommon compared with legitimate transactions. This project investigates whether transaction details—such as transaction type, amount, and account balances before and after a payment—can help identify potentially fraudulent activity.

**Goal:** Build and evaluate machine learning classifiers that estimate whether a transaction is fraudulent.

## Dataset

The project uses a transaction dataset with the following columns:

| Column | Description |
|---|---|
| `step` | Time step associated with the transaction |
| `type` | Transaction type |
| `amount` | Transaction amount |
| `nameOrig` | Identifier of the originating account |
| `oldbalanceOrg` | Origin account balance before the transaction |
| `newbalanceOrig` | Origin account balance after the transaction |
| `nameDest` | Identifier of the destination account |
| `oldbalanceDest` | Destination account balance before the transaction |
| `newbalanceDest` | Destination account balance after the transaction |
| `isFraud` | Target: `1` for fraud and `0` for non-fraud |

The supplied `new_data.csv` contains **1,048,575 transactions**, including **1,142 fraudulent transactions** and **1,047,433 non-fraudulent transactions**. The fraud class represents approximately **0.109%** of this dataset.

### Dataset files and GitHub upload size

The CSV is larger than GitHub's standard 25 MB browser-upload limit. To make it easier to store in the repository, it is split into four gzip-compressed CSV files:

- `new_data_part_1.csv.gz`
- `new_data_part_2.csv.gz`
- `new_data_part_3.csv.gz`
- `new_data_part_4.csv.gz`

Each compressed part is below 25 MB. The parts are data chunks, not separate datasets. Keep all four files together in the project directory. The `merge_dataset.py` script combines them into the original `new_data.csv` file.

**Do not rename the part files** unless you also update the merge script.

## Project Workflow

1. **Load and inspect the data** with Pandas.
2. **Perform exploratory data analysis (EDA)**, including transaction counts and amounts by type, fraud distribution, time-step distribution, and a correlation heatmap.
3. **Prepare model features**:
   - One-hot encode the `type` column with `pandas.get_dummies()` and `drop_first=True`.
   - Use `isFraud` as the target variable.
   - Remove `isFraud`, `type`, `nameOrig`, and `nameDest` from the model feature matrix.
4. **Split the data** into training and test sets using an 80:20 split and `random_state=42`.
5. **Train and compare classifiers**:
   - Logistic Regression
   - XGBoost Classifier
   - Random Forest Classifier (`n_estimators=7`, `criterion='entropy'`, `random_state=7`)
6. **Evaluate predictions** using ROC-AUC for training and test predictions.
7. **Visualize results** with a confusion matrix for the Logistic Regression model.

## Tech Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook

## Repository Structure

```text
.
├── Online Payment Fraud Detection.ipynb
├── new_data_part_1.csv.gz
├── new_data_part_2.csv.gz
├── new_data_part_3.csv.gz
├── new_data_part_4.csv.gz
├── merge_dataset.py
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Dhawal-Bhawsar/<repository-name>.git
cd <repository-name>
```

Replace `<repository-name>` with the actual name of your GitHub repository.

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
```

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
```

### 4. Rebuild the dataset

Make sure all four `.csv.gz` files and `merge_dataset.py` are in the same directory. Run:

```bash
python merge_dataset.py
```

The script reads the four compressed parts, combines their rows, and creates `new_data.csv` in the current directory. It prints the resulting row and column counts.

### 5. Run the notebook

```bash
jupyter notebook
```

Open `Online Payment Fraud Detection.ipynb` and run the cells in order. The notebook loads the dataset using:

```python
data = pd.read_csv("new_data.csv")
```

## Model Evaluation and Limitations

- **Class imbalance:** Only 1,142 of the 1,048,575 transactions in the supplied CSV are labelled as fraud (approximately 0.109%). Accuracy alone would be misleading for this task.
- **Metric naming:** The notebook prints ROC-AUC values with the label “Training accuracy” / “Validation Accuracy”. ROC-AUC is not accuracy; update those labels to avoid misrepresenting the metric.
- **No reported scores in this README:** Model performance values are not stated here because verified output scores were not available in the notebook supplied for review.
- **Data splitting:** The current train/test split does not set `stratify=y`. A stratified split is worth considering for an imbalanced target.
- **Additional metrics:** Precision, recall, F1-score, PR-AUC, and threshold-specific results are not currently reported. These metrics would help assess missed fraud cases and false alarms.
- **Confusion matrix:** The notebook generates a confusion matrix for Logistic Regression only.
- **Random Forest configuration:** The current model uses seven trees. Hyperparameter tuning and validation would be needed before drawing conclusions about its performance.
- **Correlation heatmap:** Categorical values are factorized for visualization. The resulting integer codes should not be interpreted as meaningful numeric measurements.
- **Project scope:** This is an educational/portfolio project, not a production fraud detection system. Real-world use would require robust validation, monitoring, explainability, security, and controls for changing fraud patterns.

## Potential Improvements

- Use stratified train/test splitting and cross-validation.
- Compare class-weighted models and resampling methods, applying resampling only to training data.
- Report precision, recall, F1-score, ROC-AUC, and PR-AUC.
- Tune the decision threshold based on the cost of false negatives and false positives.
- Engineer additional transaction features, including balance-change indicators.
- Use SHAP or another explainability method to investigate influential features.
- Add model comparison charts and an interactive fraud-monitoring dashboard.

## Author

**Dhawal Bhawsar**

- GitHub: [Dhawal-Bhawsar](https://github.com/Dhawal-Bhawsar)

---

If you use or redistribute the dataset, include its original source and comply with the dataset's licence and usage terms.
