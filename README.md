# Binary Classification Pipeline (Logistic Regression)

> A reproducible, end-to-end binary classification workflow in scikit-learn — preprocessing, scaling, model training, GridSearchCV tuning, and a full evaluation suite. Built to demonstrate pipeline discipline, not just a one-off model.

**Stack:** Python · scikit-learn · Pandas · NumPy · Matplotlib
**Status:** Completed
**Author:** Justin Ali · [LinkedIn](https://www.linkedin.com/in/justin-ali-data/)

---

## The problem

Fitting a logistic regression is easy. Wrapping that model in a workflow another analyst can re-run six months later and get the same answer is much harder. This project is less about a specific business outcome and more about the discipline around the model: a clean pipeline with explicit preprocessing, documented validation, and an evaluation suite a stakeholder can actually read.

## Approach

1. **Preprocessing pipeline** — All transformations (imputation, scaling, encoding) wrapped in a scikit-learn Pipeline so train/test leakage is impossible by construction.
2. **EDA** — Class balance, feature distributions, correlation with target.
3. **Feature scaling** — Standardized features so logistic regression coefficients are comparable.
4. **Modeling** — Logistic regression trained inside the pipeline; coefficients interpreted as log-odds contributions.
5. **Tuning** — GridSearchCV across regularization strength (C) and penalty type (L1 vs. L2), with cross-validated ROC-AUC as the selection criterion.
6. **Evaluation** — Confusion matrix, precision, recall, ROC curve, AUC. Each metric chosen because it tells you something a different one does not.
7. **Documentation** — Every step explained inline so a stakeholder reading the notebook understands not just *what* but *why*.

## Results

A working pipeline with cross-validated performance numbers, documented methodology, and a model that another analyst can re-train or re-tune by running one notebook top to bottom.

## What I would do next

- Wrap the trained pipeline in a joblib-serialized artifact plus a small inference script for deployment.
- Add SHAP value explanations so per-prediction reasoning is auditable.
- Compare against tree-based methods (Random Forest, gradient boosting) to confirm logistic regression is actually the right choice for this data.
- Add a calibration plot — most binary classifiers need one before their probabilities are trustworthy.

## Repo contents

```
.
├── notebooks/
└── README.md
```

## How to run

```bash
git clone https://github.com/JustinAliData/binary-classification-pipeline.git
cd binary-classification-pipeline
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

## Acknowledgments

Built as part of the **Springboard Data Science Career Track**.
