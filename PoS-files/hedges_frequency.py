import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# File paths
message_path = "harmonized_data/hawkins2020_characterizing_cued/messages.csv"
trial_path = "harmonized_data/hawkins2020_characterizing_cued/trials.csv"
paper_name = os.path.basename(os.path.dirname(message_path))

# Load data
messages_df = pd.read_csv(message_path)
trials_df = pd.read_csv(trial_path)
df = pd.merge(messages_df, trials_df[['trial_id', 'rep_num']], on='trial_id', how='left')
df = df[df["role"] == "describer"].copy()

# Define hedge words
hedge_words = {
    "maybe", "perhaps", "probably", "possibly", "seems", "i think", "i guess",
    "sort of", "kind of", "kinda", "somewhat", "a little", "not sure", "might", "could",
    "likely", "appears", "looks like", "i feel like", "i suppose"
}

# Count hedge words and word count
def count_hedges_and_words(text):
    if pd.isna(text) or not text.strip():
        return pd.Series({"hedge_count": 0, "word_count": 0})
    text = text.lower()
    hedge_count = sum(text.count(hedge) for hedge in hedge_words)
    word_count = len(re.findall(r'\b\w+\b', text))
    return pd.Series({"hedge_count": hedge_count, "word_count": word_count})

df[["hedge_count", "word_count"]] = df["text"].apply(count_hedges_and_words)

# Drop empty messages to avoid divide-by-zero
df = df[df["word_count"] > 0].copy()

# Normalize hedge count
df["hedge_rate"] = df["hedge_count"] / df["word_count"]

df.to_csv("hedge_counts.csv", index=False)

# Plot with 95% CI
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="rep_num", y="hedge_rate", errorbar="ci", marker="o")
plt.title(f"{paper_name}: Average Hedge Word Rate (per word) Over Repetitions")
plt.xlabel("Repetition Number")
plt.ylabel("Average Hedge Rate (per word)")
plt.grid(True)
plt.tight_layout()
plt.show()
