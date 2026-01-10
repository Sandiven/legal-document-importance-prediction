import pandas as pd
import numpy as np
import re

train = pd.read_csv("/kaggle/input/bash-8-0-round-2/train.csv")
test = pd.read_csv("/kaggle/input/bash-8-0-round-2/test.csv")

train.head()
train.info()

TEXT_COLS = ["Headline", "Reasoning", "Key Insights"]
LIST_COLS = ["Lead Types", "Power Mentions", "Agencies", "Tags"]

for df in [train, test]:
    df[TEXT_COLS] = df[TEXT_COLS].fillna("")
    df[LIST_COLS] = df[LIST_COLS].fillna("none")

train.isnull().sum()


import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)          # remove extra spaces
    text = re.sub(r"[^\x00-\x7F]+", " ", text) # remove weird unicode
    text = text.strip()
    return text


for df in [train, test]:
    df["clean_headline"] = df["Headline"].apply(clean_text)
    df["clean_reasoning"] = df["Reasoning"].apply(clean_text)
    df["clean_key_insights"] = df["Key Insights"].apply(clean_text)


train[["Headline", "clean_headline"]].head(3)


def parse_list_column(text):
    if text == "none":
        return []
    return [t.strip().lower() for t in text.split(";") if t.strip()]


for df in [train, test]:
    df["power_mentions_list"] = df["Power Mentions"].apply(parse_list_column)
    df["lead_types_list"] = df["Lead Types"].apply(parse_list_column)
    df["agencies_list"] = df["Agencies"].apply(parse_list_column)
    df["tags_list"] = df["Tags"].apply(parse_list_column)


train[["Power Mentions", "power_mentions_list"]].head(3)


AGENCY_MAP = {
    "department of justice": "doj",
    "dept of justice": "doj",
    "federal bureau of investigation": "fbi",
    "central intelligence agency": "cia"
}


def normalize_agencies(agency_list):
    return [AGENCY_MAP.get(a, a) for a in agency_list]


for df in [train, test]:
    df["agencies_list"] = df["agencies_list"].apply(normalize_agencies)

train[["Agencies", "agencies_list"]].head(5)


for df in [train, test]:
    df["num_power_mentions"] = df["power_mentions_list"].apply(len)
    df["num_lead_types"] = df["lead_types_list"].apply(len)
    df["num_agencies"] = df["agencies_list"].apply(len)
    df["num_tags"] = df["tags_list"].apply(len)


train[[
    "power_mentions_list", "num_power_mentions",
    "agencies_list", "num_agencies"
]].head(5)


FINAL_COLS = [
    "id",
    "clean_headline",
    "clean_reasoning",
    "clean_key_insights",
    "power_mentions_list",
    "lead_types_list",
    "agencies_list",
    "tags_list",
    "num_power_mentions",
    "num_lead_types",
    "num_agencies",
    "num_tags"
]

train_final = train[FINAL_COLS + ["Importance Score"]]
test_final = test[FINAL_COLS]

train_final.head()



train_final.to_csv("data/train_cleaned.csv", index=False)
test_final.to_csv("data/test_cleaned.csv", index=False)

