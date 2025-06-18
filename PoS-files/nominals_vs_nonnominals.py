import pandas as pd
import spacy
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")
tqdm.pandas()

# File paths
message_path = "yoon2019_audience/messages.csv"
trial_path = "yoon2019_audience/trials.csv"

# Extract paper name from path
paper_name = os.path.basename(os.path.dirname(message_path))

# Load data
messages = pd.read_csv(message_path)
trials = pd.read_csv(trial_path)

# Merge to get rep_num
df = pd.merge(messages, trials[['trial_id', 'rep_num']], on='trial_id', how='left')

# Filter to describer messages only
df = df[df["role"] == "describer"].copy()

# Function to classify PoS tokens
def classify_tokens(text):
    doc = nlp(str(text))
    nominals = [token for token in doc if token.pos_ in ["NOUN", "PROPN", "PRON"]]
    non_nominals = [token for token in doc if token.pos_ not in ["NOUN", "PROPN", "PRON"] and token.is_alpha]
    return len(nominals), len(non_nominals)

# Apply classification with progress
df[["num_nominals", "num_non_nominals"]] = df["text"].fillna("").progress_apply(
    lambda x: pd.Series(classify_tokens(x))
)

# Compute proportion nominal
df["total"] = df["num_nominals"] + df["num_non_nominals"]
df = df[df["total"] > 0]  # Avoid division by zero
df["prop_nominal"] = df["num_nominals"] / df["total"]

# Group by repetition number
agg_df = df.groupby("rep_num", as_index=False).agg({
    "prop_nominal": "mean"
})

# Add rolling average for smoothing (optional)
agg_df["rolling"] = agg_df["prop_nominal"].rolling(window=3, center=True).mean()

# Plot
plt.figure(figsize=(10, 5))
plt.plot(agg_df["rep_num"], agg_df["prop_nominal"], marker='o', label="Avg per Rep")
#plt.plot(agg_df["rep_num"], agg_df["rolling"], color="red", linestyle="--", label="Smoothed (window=3)")
plt.xlabel("Repetition Number")
plt.ylabel("Proportion Nominal (NOUN, PROPN, PRON)")
plt.title(f"{paper_name}: Nominal Usage Over Repetitions")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
