# decoding_digital_personas

**MBTI Classification via Social Media Content Analysis**

## 🔍 Overview

This project aims to predict MBTI (Myers-Briggs Type Indicator) personality types from social media posts using a combination of traditional machine learning models and transformer-based deep learning approaches.  
We analyze social media posts to find language patterns that are linked to different MBTI personality traits.

## 📚 Motivation

Understanding personality types from social media content has applications in personalized recommendations, career guidance, mental health, and human-computer interaction. 

## 🧠 Objectives

- Build models to classify MBTI types based on textual data.
- Compare traditional ML models (e.g., Logistic Regression, Random Forest) with deep learning models (e.g., LSTM, BERT).
- Use interpretable AI methods (e.g., SHAP) to explain predictions.
- Evaluate performance across multiple feature representations.

## 🗂️ Dataset

- **MBTI Dataset** from Kaggle:  
  Includes ~8600 user posts labeled with one of the 16 MBTI types.  
  Each sample contains 50 concatenated social media posts.

## ⚙️ Methodology

### ➤ Traditional Machine Learning

- **Models Used:** Logistic Regression, Random Forest, (SVM - optional)
- **Feature Representations:**
  - CountVectorizer
  - TF-IDF
  - Word2Vec / GloVe embeddings (optional extension)
- **Explainability:** SHAP (Shapley Additive Explanations)

### ➤ Deep Learning

- **Models Used:** LSTM, BERT
- **Fine-Tuning:** BERT with MLP head for classification
- **Libraries:** `transformers`, `torch`, `scikit-learn`, `pandas`, `matplotlib`

## 🧪 Experiments

- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score (multi-class)
- **Classification Setup:**
  - 16-class classification task
  - Also evaluated dimension-wise (I/E, N/S, T/F, J/P)
- **Data Split:** 70% training / 15% validation / 15% testing

## 📈 Results

- Comparative performance of ML vs. BERT models
- Interpretation of feature importance using SHAP
- Confusion matrices and classification reports for analysis


## 📄 License

This project is licensed under the MIT License.

