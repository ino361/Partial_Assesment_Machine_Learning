# Predictive Modelling for Credit Card Default Using Machine Learning

MSc Artificial Intelligence — Introduction to Artificial Intelligence (Practical skills assessment)

**Author:** Inancan Cagimni (Q1140517)
**Module lecturer:** Dr. Lawrence Ibeh

## What this repository contains

- `Credit_Default_Analysis.ipynb` — full notebook: data cleaning, EDA, data preparation, model
  training, evaluation, hyperparameter tuning, SHAP interpretation, and a leakage experiment.
- `credit_default_pipeline.py` — the same pipeline as a plain Python script.
- `Credit_Card.csv` — the dataset used (34,788 raw rows, 30,000 unique clients after cleaning).

## How to run this

1. Put `Credit_Card.csv` in the same folder as the notebook (already included here).
2. Open `Credit_Default_Analysis.ipynb` in Google Colab or Jupyter.
3. Install the requirements: `pip install pandas numpy scikit-learn matplotlib seaborn scipy shap`
4. Run all cells top to bottom. It takes about 15 minutes (hyperparameter tuning is the slow part).
5. Figures are saved to `figures/` and tables to `tables/`.

Everything is seeded with `RANDOM_STATE = 42`, so results should be identical on any machine.

## Summary of results

- Best model: **Gradient Boosting (tuned)**, AUC-ROC = 0.7775 on the held-out test set.
- The `risk_leak` column in the raw data is a leaked feature: using it gives AUC = 1.00 for every
  model, which is unrealistic. It was excluded from the final model.
- Full details, discussion and limitations are in the accompanying report
  (`Q1140517_Predictive_Modelling_Report.docx`).
