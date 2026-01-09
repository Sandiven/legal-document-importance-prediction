import os
import joblib

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from scipy.sparse import hstack
from sklearn.linear_model import Ridge

import lightgbm as lgb


# ================= CONFIG =================
TRAIN_MODEL = True          # True = retrain + validate
MAKE_SUBMISSION = True    # True = train on full data + create submission


# ================= LOAD DATA =================
train = pd.read_csv("data/train_cleaned.csv")
test  = pd.read_csv("data/test_cleaned.csv")


# ================= CLEAN TEXT =================
text_cols = [
    "clean_headline",
    "clean_reasoning",
    "clean_key_insights"
]

for col in text_cols:
    train[col] = train[col].fillna("")
    test[col]  = test[col].fillna("")


# ================= TEXT FIELDS =================
train["full_text"] = (
    train["clean_headline"] + " " +
    train["clean_key_insights"] + " " +
    train["clean_reasoning"]
)

test["full_text"] = (
    test["clean_headline"] + " " +
    test["clean_key_insights"] + " " +
    test["clean_reasoning"]
)

train["headline_text"] = train["clean_headline"]
test["headline_text"]  = test["clean_headline"]

train["reasoning_text"] = train["clean_reasoning"]
test["reasoning_text"]  = test["clean_reasoning"]


# ================= NUMERIC FEATURES =================
for df in [train, test]:
    df["word_count"] = df["full_text"].str.split().str.len()

num_cols = [
    "num_power_mentions",
    "num_lead_types",
    "num_agencies",
    "num_tags",
    "word_count"
]


# ================= SPLIT =================
X_train, X_val, y_train, y_val = train_test_split(
    train.index,
    train["Importance Score"],
    test_size=0.2,
    random_state=42
)


# ================= TF-IDF VECTORIZERS =================
tfidf_word = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=40000,
    stop_words="english"
)

tfidf_char = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=30000
)

tfidf_headline = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=15000,
    stop_words="english"
)

tfidf_reason = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=20000,
    stop_words="english"
)


# ================= VECTORIZE (TRAIN / VAL) =================
Xtr_word = tfidf_word.fit_transform(train.loc[X_train, "full_text"])
Xval_word = tfidf_word.transform(train.loc[X_val, "full_text"])

Xtr_char = tfidf_char.fit_transform(train.loc[X_train, "full_text"])
Xval_char = tfidf_char.transform(train.loc[X_val, "full_text"])

Xtr_head = tfidf_headline.fit_transform(train.loc[X_train, "headline_text"])
Xval_head = tfidf_headline.transform(train.loc[X_val, "headline_text"])

Xtr_reason = tfidf_reason.fit_transform(train.loc[X_train, "reasoning_text"])
Xval_reason = tfidf_reason.transform(train.loc[X_val, "reasoning_text"])

num_train = train.loc[X_train, num_cols]
num_val   = train.loc[X_val, num_cols]


# ================= STACK FEATURES (HEADLINE WEIGHTED) =================
X_train_final = hstack([
    Xtr_word,
    Xtr_char,
    Xtr_head,
    Xtr_head,          # headline weighting
    Xtr_reason,
    num_train
])

X_val_final = hstack([
    Xval_word,
    Xval_char,
    Xval_head,
    Xval_head,
    Xval_reason,
    num_val
])


# ================= MODEL =================
model = lgb.LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.02,
    num_leaves=96,
    min_child_samples=30,
    subsample=0.75,
    colsample_bytree=0.75,
    reg_alpha=0.3,
    reg_lambda=0.3,
    random_state=42,
    n_jobs=-1
)


# ================= TRAIN / LOAD =================
if TRAIN_MODEL:
    print("Training model...")
    model.fit(
        X_train_final, y_train,
        eval_set=[(X_val_final, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(150)]
    )

    val_preds = model.predict(X_val_final)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print("Validation RMSE:", rmse)

    joblib.dump(model, "lgb_model.pkl")
    joblib.dump(tfidf_word, "tfidf_word.pkl")
    joblib.dump(tfidf_char, "tfidf_char.pkl")
    joblib.dump(tfidf_headline, "tfidf_headline.pkl")
    joblib.dump(tfidf_reason, "tfidf_reason.pkl")

else:
    print("Loading saved model...")
    model = joblib.load("lgb_model.pkl")
    tfidf_word = joblib.load("tfidf_word.pkl")
    tfidf_char = joblib.load("tfidf_char.pkl")
    tfidf_headline = joblib.load("tfidf_headline.pkl")
    tfidf_reason = joblib.load("tfidf_reason.pkl")


# ================= FINAL TRAIN + SUBMISSION =================
if MAKE_SUBMISSION:
    print("Creating submission...")

    X_full_word = tfidf_word.fit_transform(train["full_text"])
    X_full_char = tfidf_char.fit_transform(train["full_text"])
    X_full_head = tfidf_headline.fit_transform(train["headline_text"])
    X_full_reason = tfidf_reason.fit_transform(train["reasoning_text"])

    X_test_word = tfidf_word.transform(test["full_text"])
    X_test_char = tfidf_char.transform(test["full_text"])
    X_test_head = tfidf_headline.transform(test["headline_text"])
    X_test_reason = tfidf_reason.transform(test["reasoning_text"])

    X_full_final = hstack([
        X_full_word,
        X_full_char,
        X_full_head,
        X_full_head,
        X_full_reason,
        train[num_cols]
    ])

    X_test_final = hstack([
        X_test_word,
        X_test_char,
        X_test_head,
        X_test_head,
        X_test_reason,
        test[num_cols]
    ])

    model.fit(X_full_final, train["Importance Score"])
    # ===== Train Ridge Regression =====
    ridge = Ridge(alpha=5.0)
    ridge.fit(X_full_final, train["Importance Score"])


    # ===== Predict with both models =====
    pred_lgb = model.predict(X_test_final)
    pred_ridge = ridge.predict(X_test_final)

    # ===== Ensemble (weighted average) =====
    test_preds = 0.8 * pred_lgb + 0.2 * pred_ridge
    test_preds = test_preds.clip(0, 100)


    submission = pd.DataFrame({
        "id": test["id"],
        "Importance Score": test_preds
    })

    submission.to_csv("submission.csv", index=False)
    print("submission.csv created successfully ✅")
