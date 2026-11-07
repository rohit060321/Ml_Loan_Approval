# Loan Approval Prediction

A machine learning project that explores applicant and loan-related information and builds classification models to predict whether a loan application will be approved.

## Project overview

This repository contains:
- `credit_wise.ipynb` — a Jupyter Notebook with data inspection, cleaning, exploratory data analysis (EDA), feature encoding, correlation analysis, model training, and evaluation.
- `loan_approval_data.csv` — the dataset used by the notebook.

The prediction target is **`Loan_Approved`**, with two classes: `Yes` and `No`.

## Dataset

The supplied CSV contains **1,000 rows and 20 columns**. It includes applicant demographics, income, employment, credit profile, loan details, and the approval outcome.

### Column descriptions

| Column | Description |
|---|---|
| `Applicant_ID` | Applicant identifier |
| `Applicant_Income` | Applicant's income |
| `Coapplicant_Income` | Co-applicant's income |
| `Employment_Status` | Employment status |
| `Age` | Applicant's age |
| `Marital_Status` | Marital status |
| `Dependents` | Number of dependents |
| `Credit_Score` | Applicant's credit score |
| `Existing_Loans` | Number of existing loans |
| `DTI_Ratio` | Debt-to-income ratio |
| `Savings` | Applicant's savings |
| `Collateral_Value` | Value of collateral |
| `Loan_Amount` | Requested loan amount |
| `Loan_Term` | Loan duration/term |
| `Loan_Purpose` | Purpose of the loan |
| `Property_Area` | Property location category |
| `Education_Level` | Education category |
| `Gender` | Applicant's gender |
| `Employer_Category` | Employer category |
| `Loan_Approved` | Target label: `Yes` or `No` |

### Data quality notes

- The supplied file has 1,000 records and 20 columns.
- Each column contains 50 missing values (5% of the rows per column).
- No fully duplicated rows were found in the supplied CSV.
- The notebook imputes numerical missing values using the mean and categorical missing values using the most frequent value.
- `Applicant_ID` is removed before modelling because it is an identifier rather than an applicant characteristic.

These figures describe the supplied version of the dataset; they may change if the CSV is updated.

## Workflow in the notebook

1. **Load and inspect data** — preview records, inspect data types, summary statistics, and missing values.
2. **Handle missing values** — impute numerical and categorical columns.
3. **Exploratory data analysis** — inspect the target distribution, categorical features, income distributions, and relationships between selected variables and loan approval.
4. **Outlier exploration** — use box plots to inspect numerical features.
5. **Feature preparation** — remove the applicant identifier and encode categorical variables.
6. **Correlation analysis** — calculate and visualise correlations among numerical features.
7. **Train-test split** — divide the data into training and test sets.
8. **Feature scaling** — standardise model inputs.
9. **Model training and evaluation** — train Logistic Regression, K-Nearest Neighbours (KNN), and Gaussian Naive Bayes classifiers; compare their evaluation metrics.
10. **Feature engineering** — experiment with squared versions of `DTI_Ratio` and `Credit_Score` and evaluate the models again.

## Models and evaluation

The notebook experiments with:
- Logistic Regression
- K-Nearest Neighbours (KNN)
- Gaussian Naive Bayes

Model performance is assessed using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

The notebook includes a note identifying Naive Bayes as the best model based on precision in that run. Re-run the notebook to verify the result for your installed library versions and current data.

## Requirements

Python 3.9 or later is recommended.

Install the main dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## How to run

1. Clone or download this project.
2. Place the notebook and CSV in the same directory.
3. Make sure the CSV is named `loan_approval_data.csv`, matching the path used in the notebook. Alternatively, update the `pd.read_csv(...)` path in the first data-loading cell.
4. Install the dependencies listed above.
5. Start Jupyter:

   ```bash
   jupyter notebook
   ```

6. Open `credit_wise.ipynb` and run the cells from top to bottom.

## Repository structure

```text
loan-approval-prediction/
├── credit_wise.ipynb
├── loan_approval_data.csv
└── README.md
```

## Notes and possible improvements

- **Avoid data leakage:** for a production-quality workflow, split the data before fitting imputers, encoders, or scalers. Fit preprocessing only on the training data, preferably through a scikit-learn `Pipeline` and `ColumnTransformer`.
- **Stratification:** for this classification task, consider using `stratify=y` in `train_test_split` to preserve the class proportions in both splits.
- **Encoding:** use a consistent preprocessing pipeline for categorical features. Fit encoders on training data only and configure them to handle categories not seen during training.
- **Reproducibility:** set and document random seeds, and consider cross-validation when comparing models.
- **Fairness and responsible use:** demographic attributes such as age and gender may raise legal and ethical concerns in lending. This notebook is an educational exercise, not a credit decision system. Any real-world use would require appropriate legal review, fairness testing, explainability, security, and human oversight.

## Disclaimer

This project is intended for learning and experimentation with a tabular binary-classification problem. Its predictions should not be used to make real lending or financial decisions.
