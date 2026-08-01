# Final Project — Reddit Sentiment Analysis (Israel–Palestine Discourse)

`FP_ProjectTopic_MelanieVillela.ipynb` applies the full ML workflow — EDA, text preprocessing, feature extraction, model training, and evaluation — to classify the sentiment of Reddit comments discussing the Israel–Palestine conflict.

## Business problem

Public discourse around high-profile, polarized geopolitical events shifts quickly. Political strategists, policy advisors, and community moderators need a way to track how sentiment is trending across social platforms without manually reading thousands of comments. This project builds and compares classification models that automatically label a comment as **positive**, **negative**, or **neutral**.

## Dataset

- **Source**: [Reddit on Israel-Palestine (daily updated)](https://www.kaggle.com/datasets/asaniczka/reddit-on-israel-palestine-daily-updated) via `kagglehub`.
- **Sampling**: 5,000 rows were sampled (`random_state=42`) from the source and frozen locally to `reddit_sample_frozen.csv`, since the source dataset updates daily and reproducibility requires a fixed snapshot.
- **Filtering**: After an initial review surfaced off-topic comments, the sample was filtered to four topically relevant subreddits (`IsraelPalestine`, `worldnews`, `Palestine`, `AskMiddleEast`), leaving **3,932 rows**.
- **Labeling**: Sentiment labels were generated with NLTK's VADER lexicon (compound score ≥ 0.05 → positive, ≤ -0.05 → negative, else neutral). Class distribution is imbalanced: ~47-48% negative, ~31-32% positive, ~19-21% neutral.

## Methodology

1. **EDA** — dataset structure, missing values, class balance, text length by sentiment, most common words, subreddit relevance review.
2. **Preprocessing** — lowercasing, URL/mention/hashtag stripping, punctuation removal, stopword removal (NLTK English + custom contraction remnants), then TF-IDF vectorization (unigrams + bigrams, 5,000 max features).
3. **Modeling** — three classifiers trained and compared on an 80/20 stratified train/test split:
   - Logistic Regression (`class_weight='balanced'`)
   - Multinomial Naive Bayes
   - Random Forest (`class_weight='balanced'`)
4. **Evaluation** — accuracy, weighted precision/recall/F1, confusion matrices, and manual error analysis on misclassified examples.

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Logistic Regression** | **64.29%** | **0.6596** | **0.6429** | **0.6464** |
| Random Forest | 58.45% | — | — | 0.5930 |
| Naive Bayes | 56.67% | 0.6412 | 0.5667 | 0.4835 |

Logistic Regression was the strongest model overall. Naive Bayes struggled most on the neutral class (0.02 recall), which drove its weak F1-score despite a reasonable accuracy.

## Key limitations

- Results reflect a single frozen snapshot of a daily-updated dataset — not the full ~4M-row source.
- VADER-generated labels are lexicon-based and can misread sarcasm or context-dependent hostility.
- Comments were classified in isolation, without surrounding thread/reply context.

## How to run

1. Ensure `reddit_sample_frozen.csv` is present in this folder (already frozen for reproducibility — the notebook will load it directly instead of re-downloading/re-sampling from Kaggle).
2. Open `FP_ProjectTopic_MelanieVillela.ipynb` in VS Code or Jupyter.
3. Run all cells top to bottom. First-run cells will download the NLTK `vader_lexicon` and `stopwords` resources.

## Files

- `FP_ProjectTopic_MelanieVillela.ipynb` — the full project notebook.
- `reddit_sample_frozen.csv` — frozen 5,000-row data sample used for reproducible results.
