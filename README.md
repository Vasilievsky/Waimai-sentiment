# Sentiment Classification of Chinese Food Delivery Reviews

## Goal

Classify Chinese food delivery (外卖) reviews as positive or negative using TF-IDF and classical ML models.

## Data

waimai_10k from the ChineseNlpCorpus collection (SophonPlus/ChineseNlpCorpus). 11,987 reviews, 11,980 after cleaning. Label 1 = positive, 0 = negative.

## EDA summary

- Classes are imbalanced: ~67% negative, ~33% positive.
- 7 reviews had conflicting labels; after manual reading I set them to 1 and kept one copy of each.
- Negative reviews are longer on average (~29 vs ~17 characters).

## Method

- Split: 80/20 train/test (9,584 / 2,396 reviews), stratified. The test set is used only once, for the final model.
- Features: TF-IDF on character n-grams, so no word segmentation is needed for Chinese.
- All comparisons and tuning use stratified 5-fold cross-validation on the train set only.
- Models: logistic regression, decision tree, kNN (cosine distance). Hyperparameters were tuned manually and with Optuna.

## Metric

Main metric: **macro-F1**. I first planned to use recall/F2 of the negative class (a platform wants to catch complaints), but negative reviews are the majority, so a dummy model that always predicts 0 already gets recall = 1.0 and F2 = 0.91. Macro-F1 treats both classes equally and clearly separates models from the dummy (0.40). Precision and recall of the negative class are reported alongside.

## Results

### 1. Manual comparison of logistic regression variants (5-fold CV on train)

| Model                                       | macro-F1 | std   | recall (class 0) | precision (class 0) |
| ------------------------------------------- | -------- | ----- | ---------------- | ------------------- |
| Dummy (always 0)                            | 0.400    | -     | 1.000            | 0.666               |
| Logistic regression, C=10                   | 0.887    | 0.007 | 0.950            | 0.907               |
| Logistic regression, balanced class weights | 0.883    | 0.008 | 0.920            | 0.923               |
| CountVectorizer instead of TF-IDF           | 0.878    | 0.009 | 0.913            | 0.922               |
| Baseline (TF-IDF 1-2 grams, C=1)            | 0.877    | 0.010 | 0.962            | 0.890               |
| TF-IDF 1-3 grams                            | 0.872    | 0.010 | 0.964            | 0.884               |
| CountVectorizer + C=10                      | 0.871    | 0.008 | 0.922            | 0.908               |
| Logistic regression, C=0.1                  | 0.781    | 0.010 | 0.980            | 0.807               |

- All reasonable variants lie within ~0.015 macro-F1, and the top ones differ by less than the fold-to-fold std.
- Weak regularization (C=0.1) is clearly worse; larger C helps.
- Class weights trade recall of complaints for precision.
- CountVectorizer instead of TF-IDF did not help.

### 2. Hyperparameter tuning with Optuna (logistic regression)

30 trials over C (0.5-100, log scale), class weights and n-gram range. All of the top 10 trials use balanced class weights and 1-3 grams, and C anywhere in ~4-20 gives almost the same score (CV macro-F1 ≈ 0.889-0.891). The gain over the manual C=10 variant is ~0.004, within the fold-to-fold std. I fixed C=10 for the final model.

### 3. Model comparison (5-fold CV on train)

| Model                                           | macro-F1 | std   | recall (class 0) | precision (class 0) |
| ----------------------------------------------- | -------- | ----- | ---------------- | ------------------- |
| Logistic regression (C=10, balanced, 1-3 grams) | 0.890    | 0.008 | 0.936            | 0.920               |
| kNN (k=30, distance weights, Optuna)            | 0.846    | 0.014 | 0.919            | 0.884               |
| Decision tree (depth 11, Optuna)                | 0.820    | 0.008 | 0.887            | 0.876               |

Logistic regression is clearly the best: its advantage (~0.04 over kNN, ~0.07 over the tree) is much larger than the CV std. The tree overfits on thousands of sparse features, and kNN suffers from the majority class dominating the neighbors at large k. Final model: **logistic regression, TF-IDF on character 1-3 grams, C=10, balanced class weights.**

### 4. Final evaluation on the held-out test set

Macro-F1 = **0.886**, accuracy = **0.90** (dummy: 0.40 and 0.67). This is consistent with the CV estimate (0.889 ± 0.008).

| Class        | precision | recall | F1    |
| ------------ | --------- | ------ | ----- |
| negative (0) | 0.909     | 0.944  | 0.927 |
| positive (1) | 0.880     | 0.812  | 0.845 |

The model makes 150 false alarms (positive reviews predicted as negative) and misses 89 complaints.

## Error analysis

I read 30 random misclassified reviews (15 false alarms and 15 missed complaints).

- About 13 of 30 look like **label errors** (e.g. a clear complaint labeled positive, or "delivery was on time" labeled negative). In one review the author writes that they habitually give good ratings anyway, so the labels may partly come from star ratings and not from the text.
- About 6 are **mixed reviews** (praise plus a complaint about delivery speed or temperature).
- About 10 are **real model errors**, mostly implicit criticism without typical sentiment words and negation.
- The most confident errors are mostly label problems.

The test macro-F1 therefore probably underestimates the model's quality on the text itself. This is based on a small sample read by one person.

## Limitations and future work

- Noisy labels: cleaning them (or re-labeling a sample) would give a more reliable evaluation.
- Character n-grams handle negation and implicit criticism only partially. Word-level features or a pretrained Chinese language model could help.
- Mixed reviews might need a third "neutral" class.
- The decision threshold could be tuned with cross-validation on train, for example to favor recall of complaints.

## Status

- [x] EDA
- [x] Baseline and comparison of variants
- [x] Hyperparameter tuning with Optuna
- [x] Decision tree and kNN for comparison
- [x] Final evaluation on the test set
- [x] Error analysis

## Project structure

- `data/` — dataset (not included, see Data)
- `notebook/`
  - `01_eda.ipynb` — exploratory analysis
  - `02_baseline.ipynb` — split, baseline, comparison of variants
  - `03_tuning.ipynb` — Optuna tuning, tree and kNN, final model choice
  - `04_final_evaluation.ipynb` — test evaluation and error analysis
- `requirements.txt` — dependencies

## How to run

```
python -m venv waimai-env
source waimai-env/bin/activate
pip install -r requirements.txt
```

Download waimai_10k from the ChineseNlpCorpus repository, put it into `data/` and run the notebooks in order.
