# decoding_digital_personas

**MBTI Classification via Social Media Content Analysis**

## Overview

This project explores the task of predicting MBTI (Myers-Briggs Type Indicator) personality types from social media posts. We compare traditional machine learning models (e.g., Logistic Regression, XGBoost) with a transformer-based deep learning model (BERT + MLP) to classify users across four MBTI trait dimensions: I/E, N/S, F/T, and P/J.

The goal is to evaluate which models are most effective at identifying personality traits from natural language and to explore interpretability and performance trade-offs.

## How to Run the Project

### Prerequisites
- Python 3.8+
- PyTorch
- scikit-learn
- pandas
- transformers (Hugging Face)
- matplotlib / seaborn (for visualization)

### 🔧 Clone and Install

```bash
git clone https://github.com/xyzesther/decoding_digital_personas.git
cd decoding_digital_personas
pip install -r requirements.txt
```

## Methodology

### ➤ Traditional Machine Learning

- **Models Used:** Logistic Regression, XGBoost
- **Preprocessing**: TF-IDF on cleaned posts (lowercase, remove URLs, HTML, emojis,
punctuation, stopwords) → top 3000 features
- **Feature Representations:**
  - CountVectorizer
  - TF-IDF
- **Model Setup for XGBoost**
  - **Architecture**: XGBoost classifier (100 trees, max depth 5, learning rate 0.1)
  - **Output**: 4 binary classifiers (I/E, N/S, F/T, P/J) trained independently
  - **Class Imbalance**: scale_pos_weight used based on training data distribution
  - **Evaluation**: Accuracy + Classification Report + Confusion Matrix
  - **Visualization**: Training log loss curve for convergence tracking
    
- **Model Setup for Logistic Regression**
  - **Architecture**: Scikit-learn LogisticRegression (max_iter=1000, solver='saga')
  - **Output**: 4 binary classifiers (I/E, N/S, F/T, P/J) trained independently
  - **Class Imbalance**: class_weight='balanced' used for I/E and N/S traits
  - **Evaluation**: Accuracy + Classification Report + Confusion Matrix
  - **Visualization**: Training log loss curve for convergence tracking
  - **Interpretability**: Top positive/negative word features visualized by coefficient weight
### ➤ Deep Learning

- **Model Used:** BERT with a multi-layer perceptron (MLP) classifier head
- **Approach:**
  - Fine-tune the full BERT model with a small MLP on top for 4-way binary MBTI trait prediction (I/E, N/S, T/F, J/P)
  - Training on concatenated social media posts (first 256 tokens per user)
- **Architecture:**
  - Pretrained BERT-base (bert-base-uncased)
  - MLP with hidden layer: 768 → 50 → 4
- **Loss Function:** BCEWithLogitsLoss for multi-label binary classification
- **Training Framework:** PyTorch
- **Tokenizer:** Hugging Face transformers.BertTokenizer
- **Evaluation:** Predictions converted to 4 binary labels, mapped to 16 MBTI types

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xyzesther/decoding_digital_personas/blob/main/BERT_MLP_Model_MBTI_Prediction.ipynb)


## Expected Output

- After running the scripts, you will get:
- Data distribution of the dataset
- Classification reports and Confusion matrices for all 4 MBTI traits (precision, recall, F1)


### Summary Results
| Model             | Macro F1 | Best Trait |
|------------------|----------|------------|
| Logistic Regression | 0.79     | F/T        |
| XGBoost             | 0.77     | N/S        |
| BERT + MLP          | 0.71     | F/T, P/J   |

Traditional models achieved higher overall macro F1 scores, while BERT performed well on traits with subtle contextual cues (e.g., F/T and P/J).


## 📄 License

This project is licensed under the MIT License.

