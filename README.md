# 🏦 Credit Risk Default Prediction with Explainable AI

## 📖 Overview

When a bank gives out a loan, it takes on financial risk. If the borrower doesn't repay the loan, the bank loses money. However, if the bank is too strict and rejects too many applicants, it loses potential business.

This project uses **Machine Learning to predict the probability of loan default before a loan is approved**.

Beyond prediction, the project also focuses on **Explainable AI (XAI)** using **SHAP**, allowing the model's predictions to be interpreted at both the global and individual applicant levels.

---

## 📊 Dataset

I used the **Lending Club Loan Data** dataset, which contains real-world loan applications, borrower credit histories, and final loan outcomes such as:

* `Fully Paid`
* `Charged Off`

Link - https://www.kaggle.com/datasets/wordsforthewise/lending-club

The raw dataset contains approximately **151 columns** with mixed data types, including:

* Numerical features
* Dates
* Categorical variables
* Text-based fields
* Missing values

---

# 🛠️ Step-by-Step Breakdown

## Step 1: Data Cleaning & Removing Empty Columns

Machine Learning models cannot directly handle completely empty features.

The raw dataset contained many columns that were either completely empty or had a very high percentage of missing values.

I wrote a preprocessing pipeline that automatically identifies and removes columns that are **100% null**, reducing unnecessary features and simplifying the dataset.

---

## Step 2: Preventing Data Leakage

One of the most important parts of the project was preventing **data leakage**.

In a real loan application, the model can only use information available **at the time the loan is issued**.

However, the original dataset contained features that were generated **after loan origination**, such as:

* Total late fees paid
* Last payment amount
* Subsequent payment information
* Other post-origination variables

Using these features would allow the model to indirectly "look into the future."

This would produce artificially high performance and make the model unsuitable for real-world deployment.

I therefore identified and removed **20+ post-origination features** to ensure that the model only uses information that would realistically be available when making the lending decision.

---

## Step 3: Feature Engineering

Machine Learning models require numerical and structured inputs, so several raw features needed to be transformed.

### Date Features

For example, instead of directly using `earliest_cr_line`, I converted it into a more meaningful feature:

> **Years of Credit History**

### String Features

Values such as:

```text
36 months
60 months
```

were converted into numerical values:

```text
36
60
```

### Categorical Features

Features such as:

* `Grade`
* `Purpose`
* Other categorical variables

were converted into categorical data types.

I used **LightGBM's native categorical feature support** instead of manually applying one-hot encoding to every categorical variable.

This reduces unnecessary dimensionality while allowing LightGBM to learn relationships between categorical values.

---

## Step 4: Chronological Train/Test Split

Instead of using a completely random train/test split, I used a **chronological split** based on the loan issue date.

The data was first sorted by:

```text
issue_d
```

The model was then trained on **older loans** and evaluated on **newer loans**.

This better simulates a real-world scenario:

```text
Past Data → Model Training → Future Data → Model Testing
```

This approach helps prevent information from future periods from influencing model training and provides a more realistic estimate of how the model may perform after deployment.

---

## Step 5: Training LightGBM & Handling Class Imbalance

Loan default is an imbalanced classification problem.

Approximately:

* **81%** of loans were non-defaults
* **19%** of loans were defaults

If a model simply predicted every loan as "non-default," it could achieve around 81% accuracy without actually identifying risky borrowers effectively.

To address this, I used **LightGBM** with:

```python
scale_pos_weight
```

This assigns greater importance to the minority default class during training.

I also used **Early Stopping** to prevent unnecessary training once validation performance stopped improving, helping reduce overfitting.

---

## Step 6: Probability Calibration

The class weighting improved the model's ability to identify risky borrowers, but it also affected the raw probability estimates.

For example, the model could become overly conservative and produce inflated default probabilities such as:

```text
43%
40%
38%
```

even when the actual observed default rate was much lower.

This creates a problem because in credit risk, the **probability itself matters**, not just the ranking of borrowers.

I therefore used **Isotonic Regression Calibration**.

The calibration process maps the model's raw predictions to probabilities that better correspond to observed outcomes.

For example:

```text
Raw Model Probability
        ↓
Isotonic Regression
        ↓
Calibrated Probability
```

The goal is that when the model predicts a **10% probability of default**, approximately 10% of similar borrowers should actually default over the relevant outcome period.

---

# 🔍 Step 7: Explainable AI with SHAP

A major component of this project is **Explainable AI**.

I used **SHAP (SHapley Additive exPlanations)** to understand how individual features influence model predictions.

SHAP is based on concepts from **game theory** and assigns contribution values to individual features.

For example, an individual prediction could be explained conceptually as:

```text
Predicted Default Risk: 30%

Debt-to-Income Ratio       → Increased risk
Recent Credit Inquiries    → Increased risk
Strong Credit Grade       → Reduced risk
Long Credit History       → Reduced risk
```

I generated both:

### Global Explanations

These show which features are generally the most important across the entire dataset.

### Local Explanations

These explain why the model produced a particular prediction for an individual applicant.

I used **SHAP Waterfall plots** to visualize these individual contributions.

---

# 🚀 Step 8: Deployment with Streamlit

A Jupyter Notebook is useful for experimentation, but the final model should be accessible through an application.

I built an interactive **Streamlit web application**.

Users can enter or adjust applicant information such as:

* Income
* Loan Amount
* Interest Rate
* Credit Grade

The application then:

1. Processes the applicant's inputs
2. Generates the model prediction
3. Applies probability calibration
4. Displays the predicted probability of default
5. Generates a SHAP explanation
6. Displays a SHAP Waterfall chart

### Application Flow

```text
User Input
    ↓
Feature Preprocessing
    ↓
LightGBM Model
    ↓
Raw Default Probability
    ↓
Probability Calibration
    ↓
Final Default Probability
    ↓
SHAP Explanation
    ↓
Streamlit Dashboard
```

---

# 📈 Results & Evaluation

Because the dataset is imbalanced, **accuracy alone is not an appropriate evaluation metric**.

I therefore evaluated the model using metrics more relevant to imbalanced credit-risk classification.

### KS Statistic

**KS Statistic: 0.36**

The **Kolmogorov-Smirnov (KS) statistic** measures the maximum separation between the score distributions of defaulted and non-defaulted borrowers.

A higher KS generally indicates better separation between the two groups.

### PR-AUC

**PR-AUC: 0.45**

The **Precision-Recall Area Under the Curve (PR-AUC)** is particularly useful for evaluating performance on the minority class.

It measures the model's ability to identify default cases while considering the trade-off between precision and recall.

---

# 💻 Tech Stack

| Category             | Technology             |
| -------------------- | ---------------------- |
| Programming Language | Python                 |
| Data Manipulation    | Pandas, NumPy          |
| Machine Learning     | LightGBM, Scikit-Learn |
| Explainable AI       | SHAP                   |
| Visualization        | Matplotlib             |
| Deployment           | Streamlit              |

---

# 📂 Project Pipeline

```text
Raw Lending Club Dataset
          ↓
Data Cleaning
          ↓
Remove Empty / High-Missing Features
          ↓
Remove Post-Origination Features
          ↓
Feature Engineering
          ↓
Categorical Feature Processing
          ↓
Chronological Train/Test Split
          ↓
LightGBM Model
          ↓
Class Imbalance Handling
          ↓
Probability Calibration
          ↓
SHAP Explainability
          ↓
Streamlit Deployment
```

---

# 🚀 How to Run the App Locally

## 1. Clone the Repository

```bash
git clone https://github.com/Shahidd7/credit-risk-prediction
cd https://github.com/Shahidd7/credit-risk-prediction
```

## 2. Install Dependencies

```bash
pip install streamlit lightgbm shap scikit-learn pandas numpy matplotlib
```

## 3. Run the Streamlit Application

```bash
streamlit run app.py
```

## 4. Open the Application

After running the command, open:

```text
http://localhost:8501
```

---

# 🎯 Key Highlights

* Built an end-to-end **credit risk prediction pipeline**
* Worked with a real-world Lending Club dataset
* Removed **data leakage** from post-origination features
* Performed feature engineering on dates, strings, and categorical variables
* Used **LightGBM** for classification
* Addressed class imbalance using `scale_pos_weight`
* Used **chronological validation** to simulate future predictions
* Calibrated predicted probabilities using **Isotonic Regression**
* Implemented **SHAP-based explainability**
* Created global and individual prediction explanations
* Deployed the model using **Streamlit**
* Evaluated performance using **KS Statistic and PR-AUC**
