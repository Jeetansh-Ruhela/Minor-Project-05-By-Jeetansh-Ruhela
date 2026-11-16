# Minor-Project-05-By-Jeetansh-Ruhela
# Credit Card Customer Churn Prediction

Machine learning minor project predicting whether bank credit card customers are existing or attrited.

## Data
- Kaggle **Credit Card Customers** dataset (`BankChurners.csv`)
- 10,127 rows, 23 original columns; 1,627 attrited customers (16.07%)
- Removed `CLIENTNUM` and two precomputed Naive Bayes columns

## Approach
1. Dataset inspection and exploratory analysis
2. Outlier detection (retained plausible values)
3. Stratified 80/20 train–test split and one-hot encoding inside sklearn pipelines
4. AdaBoost and Random Forest classification
5. GridSearchCV tuning of Random Forest (3 folds, 24 combinations)
6. Final test evaluation and feature-importance analysis

## Test results

| Model | Accuracy | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: | ---: |
| AdaBoost | 0.9358 | 0.9295 | 0.6492 | 0.7645 |
| Random Forest | 0.9477 | 0.9261 | 0.7323 | 0.8179 |
| Tuned Random Forest | **0.9541** | 0.8946 | **0.8092** | **0.8498** |

Positive class: Attrited Customer. Tuned model confusion matrix: `[[1670,31],[62,263]]`.

## Files and reproduction
- Notebook: `Minor_Project_05(2).ipynb`
- Data file required: `BankChurners.csv` (from Kaggle)
- Saved pipeline: `credit_card_churn_model.pkl`

Open the notebook in Google Colab, upload `BankChurners.csv` to `/content/`, then run the notebook top to bottom. The final pipeline includes preprocessing and prediction. A saved `.pkl` should be loaded only from a trusted source and with compatible Python/scikit-learn versions.

## Limitations
Imbalanced classes; validation on a single held-out test set; potential performance changes across time and customer populations. Feature importance does not establish causality.
