import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# File paths
message_path = "harmonized_data/hawkins2020_characterizing_cued/messages.csv"
trial_path = "harmonized_data/hawkins2020_characterizing_cued/trials.csv"
paper_name = os.path.basename(os.path.dirname(message_path))

# Load data
messages_df = pd.read_csv(message_path)
trials_df = pd.read_csv(trial_path)

# Merge to get rep_num
df = pd.merge(messages_df, trials_df[['trial_id', 'rep_num']], on='trial_id', how='left')

# Filter to describer messages only
df = df[df["role"] == "describer"].copy()

# Define function to get negation count and word count
def get_negation_and_word_count(text):
    if pd.isna(text):
        return pd.Series({"negation_count": 0, "word_count": 0})
    doc = nlp(text)
    negs = sum(1 for token in doc if token.dep_ == "neg")
    words = sum(1 for token in doc if token.is_alpha)
    return pd.Series({"negation_count": negs, "word_count": words})

# Apply to each message
df[["negation_count", "word_count"]] = df["text"].apply(get_negation_and_word_count)

# Avoid divide-by-zero errors
df = df[df["word_count"] > 0].copy()

# Calculate negation rate per message
df["negation_rate"] = df["negation_count"] / df["word_count"]

# Group by repetition and get average negation rate
neg_by_rep = df.groupby("rep_num")["negation_rate"].mean().reset_index()

# Plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=neg_by_rep, x="rep_num", y="negation_rate", marker="o")
plt.title(f"{paper_name}: Average Negation Rate per Word Over Repetitions")
plt.xlabel("Repetition Number")
plt.ylabel("Average Negation Rate (Negations per Word)")
plt.grid(True)
plt.tight_layout()
plt.show()
