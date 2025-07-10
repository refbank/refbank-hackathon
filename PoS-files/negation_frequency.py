import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# File paths
message_path = "harmonized_data/hawkins2020_characterizing_uncued/messages.csv"
trial_path = "harmonized_data/hawkins2020_characterizing_uncued/trials.csv"
paper_name = os.path.basename(os.path.dirname(message_path))

# Load data
messages_df = pd.read_csv(message_path)
trials_df = pd.read_csv(trial_path)

# Merge to get rep_num
df = pd.merge(messages_df, trials_df[['trial_id', 'rep_num']], on='trial_id', how='left')

# Filter to describer messages only
df = df[df["role"] == "describer"].copy()

# Define a function to count negation tokens using spaCy
def count_negations(text):
    if pd.isna(text):
        return 0
    doc = nlp(text)
    return sum(1 for token in doc if token.dep_ == "neg")

# Apply function
df["negation_count"] = df["text"].apply(count_negations)

# Average negation count per rep_num
neg_by_rep = df.groupby("rep_num")["negation_count"].mean().reset_index()

# Plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=neg_by_rep, x="rep_num", y="negation_count", marker="o")
plt.title(f"{paper_name}: Average Number of Negations per Message Over Repetitions")
plt.xlabel("Repetition Number")
plt.ylabel("Average Negation Count per Message")
plt.grid(True)
plt.tight_layout()
plt.show()