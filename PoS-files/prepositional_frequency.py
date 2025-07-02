import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import spacy

# File paths 
message_path = "harmonized_data/hawkins2019_continual/messages.csv"
trial_path = "harmonized_data/hawkins2019_continual/trials.csv"
paper_name = os.path.basename(os.path.dirname(message_path))

# Load data 
messages_df = pd.read_csv(message_path)
trials_df = pd.read_csv(trial_path)
df = pd.merge(messages_df, trials_df[['trial_id', 'rep_num']], on='trial_id', how='left')
df = df[df["role"] == "describer"].copy()

# Load spaCy 
nlp = spacy.load("en_core_web_sm")

# Count prepositional phrases per message 
def count_preps(text):
    if pd.isna(text) or not text.strip():
        return 0
    doc = nlp(text)
    return sum(1 for token in doc if token.dep_ == "prep")

df["prep_count"] = df["text"].apply(count_preps)

# Aggregate over repetitions 
prep_by_rep = df.groupby("rep_num")["prep_count"].mean().reset_index()

# Plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=prep_by_rep, x="rep_num", y="prep_count", marker="o")
plt.title(f"{paper_name}: Average Prepositional Phrase Count Over Repetitions")
plt.xlabel("Repetition Number")
plt.ylabel("Average Number of Prepositional Phrases per Message")
plt.grid(True)
plt.tight_layout()
plt.show()
