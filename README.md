<br />
<div align="center">
  <h1> Fake News Detection System</h1>
  <p align="center">
    An ML-powered project to classify news articles as real or fake using Logistic Regression and NLP techniques.
  </p>
</div>

---

## About the Project

**FakeNewsPrediction** is a machine learning-based classification system designed to automatically detect whether a news article is fake or real. The project leverages **Natural Language Processing (NLP)** techniques and a **Logistic Regression** model built using **Scikit-learn**, combined with robust **data preprocessing** and **feature engineering** workflows.

This project addresses a pressing real-world issue—misinformation—by providing a simple, explainable, and efficient approach to content verification.

---

##  Key Features

-  **Binary Classification**: Predicts whether an article is "Fake" or "Real".
-  **TF-IDF Vectorization**: Converts text to meaningful numerical features.
-  **Logistic Regression**: Interpretable ML model used for classification.
-  **Robust Preprocessing**: Cleans, tokenizes, and normalizes textual data.
-  **Evaluation Metrics**: Accuracy, precision, recall, and F1 score tracking.
-  **Jupyter Notebook Integration**: Easy to test, visualize, and experiment.

---

##  Technologies Used

-  Python
-  Scikit-learn
-  Logistic Regression
-  Natural Language Processing (NLP)
-  Data Preprocessing & Cleaning
-  TF-IDF (Term Frequency–Inverse Document Frequency)
-  Pandas / NumPy / Matplotlib

---

## How It Works
- Load dataset of real and fake news articles.

- Preprocess the text:

- Lowercasing, punctuation removal, stopword removal, stemming.

- Extract features using TF-IDF vectorization.

- Train a Logistic Regression model on the vectorized dataset.

- Evaluate model using accuracy, confusion matrix, and F1-score.

- Test it on custom input or unseen data.

## Example Result
Sample prediction:

- "Breaking: Scientists discover a new cure for cancer."
 Prediction: Real

- "Government bans oxygen because it's a mind control drug."
   Prediction: Fake

## Model Performance

| Metric    | Score   |
|-----------|---------|
| Accuracy  | 94.6%   |
| Precision | 93.2%   |
| Recall    | 95.1%   |
| F1 Score  | 94.1%   |

## Contact
Email - deepthidornala@gmail.com
