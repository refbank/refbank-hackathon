import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import spacy

# File paths 
message_path = "harmonized_data/hawkins2020_characterizing_cued/messages.csv"
trial_path = "harmonized_data/hawkins2020_characterizing_cued/trials.csv"
paper_name = os.path.basename(os.path.dirname(message_path))

# Load data 
messages_df = pd.read_csv(message_path)
trials_df = pd.read_csv(trial_path)
df = pd.merge(messages_df, trials_df[['trial_id', 'rep_num']], on='trial_id', how='left')
df = df[df["role"] == "describer"].copy()

# Load spaCy 
nlp = spacy.load("en_core_web_sm")

# Count prepositions and words per message
def count_preps_and_words(text):
    if pd.isna(text) or not text.strip():
        return pd.Series({"prep_count": 0, "word_count": 0})
    doc = nlp(text)
    prep_count = sum(1 for token in doc if token.dep_ == "prep")
    word_count = sum(1 for token in doc if token.is_alpha)
    return pd.Series({"prep_count": prep_count, "word_count": word_count})

df[["prep_count", "word_count"]] = df["text"].apply(count_preps_and_words)

# Filter out empty messages
df = df[df["word_count"] > 0].copy()

# Normalize
df["prep_rate"] = df["prep_count"] / df["word_count"]

# Aggregate over repetitions 
prep_by_rep = df.groupby("rep_num")["prep_rate"].mean().reset_index()

# Plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=prep_by_rep, x="rep_num", y="prep_rate", marker="o")
plt.title(f"{paper_name}: Prepositional Phrase Rate per Word Over Repetitions")
plt.xlabel("Repetition Number")
plt.ylabel("Avg. Prepositional Phrase Rate (per word)")
plt.grid(True)
plt.tight_layout()
plt.show()
