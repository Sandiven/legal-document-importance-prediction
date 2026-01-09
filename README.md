# Legal Document Importance Prediction  
*(NLP-based Regression | Hackathon Project)*

## 📌 Overview
This project focuses on building a Machine Learning system to **predict the investigative importance of legal documents**.  
The goal is to automatically assign an **Importance Score (0–100)** to each document, helping investigators and journalists prioritize high-value evidence within large document releases.

The project was developed as part of a **data science hackathon**, using a processed dataset derived from legal case documents.

---

## 🧠 Problem Statement
Large-scale legal disclosures often contain thousands of unstructured documents, many of which are administrative or low-value.  
Manually identifying critical documents is time-consuming and error-prone.

This project aims to:
- Convert unstructured legal text into meaningful numerical features
- Train a regression model to predict document importance
- Act as an **AI-assisted triage tool** for investigators

---

## 📂 Dataset Description
The dataset contains preprocessed features extracted from legal documents, including:

### Text Fields
- `clean_headline`
- `clean_reasoning`
- `clean_key_insights`

### Metadata / Engineered Features
- Number of power mentions
- Number of agencies involved
- Number of lead types
- Number of tags
- Text length and word count

> ⚠️ The dataset is not included in this repository due to size and licensing restrictions.  
> Please download it from the hackathon/Kaggle page and place it inside the `data/` directory.

---

## ⚙️ Methodology

### 1. Text Representation
To capture both frequency and structural patterns in legal language:
- **TF-IDF (word n-grams)** — captures important keywords and phrases
- **TF-IDF (character n-grams)** — captures names, abbreviations, and formatting patterns
- **Headline-specific TF-IDF (weighted)** — headlines were given higher importance due to their strong signal

### 2. Feature Engineering
Additional numeric features were added:
- Counts of agencies, power mentions, and tags
- Word count of combined text

These were concatenated with TF-IDF features to provide both semantic and structural context.

---

## 🤖 Models Used

### Primary Model
- **LightGBM Regressor**
  - Handles high-dimensional sparse features efficiently
  - Regularized to avoid overfitting
  - Early stopping based on validation RMSE

### Ensemble Model
- **Ridge Regression**
  - Trained on the same feature space
  - Provides stable, linear predictions
  - Combined with LightGBM to improve generalization

### Final Prediction
A weighted ensemble was used:
Final Prediction = 0.8 × LightGBM + 0.2 × Ridge


---

## 📊 Evaluation
- **Metric:** Root Mean Squared Error (RMSE)
- Validation RMSE closely matched leaderboard RMSE, indicating strong generalization
- Final public leaderboard score: **~3.94**
- Achieved **Top-5 ranking** on the public leaderboard

This consistency suggests the model avoids data leakage and overfitting.

---

## 🧪 Key Challenges
- Handling high-dimensional sparse text features efficiently
- Preventing feature leakage during iterative experimentation
- Managing long training times for large TF-IDF matrices
- Balancing model complexity with generalization

---

## 📈 Results & Learnings
- Feature weighting (especially headlines) significantly improved performance
- Ensembling diverse models reduced prediction variance
- Clean validation practices are crucial in leaderboard-based competitions
- Simple models combined thoughtfully can outperform complex standalone approaches

---
## ▶️ How to Run

### 1. Setup
```bash
pip install -r requirements.txt
```
### 2. Data
Place the following files inside the data/ folder:
- `train_cleaned.csv`
- `test_cleaned.csv`

### 3. Train & Predict
```bash
    python model.py
```
# This will:
- Train the model (if enabled)
- Generate predictions
- Create submission.csv

----

### 📁 Repository Structure
.
├── model.py
├── original_code.py
├── README.md
├── .gitignore
└── data/
    └── README.md

### 🔮 Future Improvements
- Incorporate sentence-level embeddings (e.g., SBERT / MiniLM)
- Apply k-fold cross-validation for more robust evaluation
- Explore stacking multiple regressors
- Add explainability dashboards (SHAP)

### 🏁 Conclusion
- This project demonstrates an end-to-end NLP regression pipeline built with a focus on clean methodology, explainability, and generalization.
- Rather than leaderboard exploitation, the emphasis was on building a model that would remain reliable on unseen data — a key requirement for real-world investigative applications.