# Loan Approval Prediction and Eligibility Analysis

This project develops machine learning models to predict loan approval status and determine eligible loan amounts and durations for applicants using client loan data.

---

## Project Overview

The goal is to:

- Predict whether a loan application will be approved (classification).
- Predict the maximum eligible loan amount if the requested amount is denied (regression).
- Predict the minimum loan duration required for approval when an applicant requests a shorter term and is denied (regression).

---

## Dataset

- `training_set.csv`: Training data with loan application features and loan status.
- `testing_set.csv`: Testing data without loan status for prediction.
- Outputs:
  - `final_submission_classification.csv`: Predicted loan approval status.
  - `final_submission_regression1.csv`: Predicted eligible loan amounts for denied applicants.
  - `final_submission_regression2.csv`: Predicted adjusted loan durations for denied applicants with short requested terms.

### Key Features

- Gender, Marital Status, Education, Dependents
- Applicant and Co-Applicant Income
- Loan Amount and Loan Term
- Credit History
- Property Area
- Loan Status (target in training data)

---

## Tasks

### 1. Loan Approval Prediction (Classification)

- Model: Random Forest Classifier
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score

### 2. Loan Amount Prediction (Regression)

- Model: Gradient Boosting Regressor
- Evaluation Metrics: RMSE, MAE, R2 score

### 3. Loan Duration Adjustment (Regression)

- Model: Gradient Boosting Regressor
- Evaluation Metrics: RMSE, MAE, R2 score

---

## Methodology

- Data preprocessing: handling missing values, encoding categorical variables, outlier capping
- Feature engineering: total income, loan-to-income ratio, EMI, balance income, one-hot encoding for `property_Area`
- Exploratory data analysis: distributions, correlations, and bivariate analyses
- Model training and validation using train-test splits
- Prediction generation and evaluation on validation and test sets

---

## Requirements

- Python 3.x
- Libraries: numpy, pandas, matplotlib, seaborn, plotly, scikit-learn

Install dependencies:
```
pip install numpy pandas matplotlib seaborn plotly scikit-learn
```


---

## Usage

1. Place `training_set.csv` and `testing_set.csv` in the working directory.
2. Run `Approval Prediction.py` to preprocess data, build and validate models, and generate predictions.
3. Find predictions in:
   - `final_submission_classification.csv`
   - `final_submission_regression1.csv`
   - `final_submission_regression2.csv`

---

## Results

- The classification model effectively predicts loan approvals with balanced metrics.
- Regression models estimate eligible loan amounts and required loan durations with acceptable error margins.
- Visualizations including bar plots and heatmaps provide insights into key loan approval factors.

---

## Project Files

- `Approval Prediction.py`: Full preprocessing, EDA, model training, and prediction pipeline.
- Dataset files: `training_set.csv`, `testing_set.csv`.
- Submission files: `final_submission_classification.csv`, `final_submission_regression1.csv`, `final_submission_regression2.csv`.

---

## License

MIT

---

## Credits

Built by [Reddy Jaideep](https://github.com/Jaideep718).



