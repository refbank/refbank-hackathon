import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# File paths
message_path = "harmonized_data/hawkins2019_continual/messages.csv"
trial_path = "harmonized_data/hawkins2019_continual/trials.csv"

# Extract paper name
paper_name = os.path.basename(os.path.dirname(message_path))

# Load data
messages_df = pd.read_csv(message_path)
trials_df = pd.read_csv(trial_path)
df = pd.merge(messages_df, trials_df[['trial_id', 'rep_num']], on='trial_id', how='left')

# Filter to describer messages only
df = df[df["role"] == "describer"].copy()

# Count definite and indefinite articles 
def count_articles(text):
    if pd.isna(text):
        return 0, 0
    doc = nlp(text.lower())
    definite = sum(1 for token in doc if token.text == "the" and token.tag_ == "DT")
    indefinite = sum(1 for token in doc if token.text in {"a", "an"} and token.tag_ == "DT")
    return definite, indefinite

# Apply
df[["definite_count", "indefinite_count"]] = df["text"].apply(
    lambda x: pd.Series(count_articles(x))
)

# Aggregate by rep_num 
article_by_rep = df.groupby("rep_num")[["definite_count", "indefinite_count"]].mean().reset_index()

# Plot 
plt.figure(figsize=(10, 6))
sns.lineplot(data=article_by_rep, x="rep_num", y="definite_count", marker="o", label="Avg. # of 'the'")
sns.lineplot(data=article_by_rep, x="rep_num", y="indefinite_count", marker="o", label="Avg. # of 'a/an'")

plt.title(f"{paper_name}: Definiteness Over Repetitions")
plt.xlabel("Repetition Number")
plt.ylabel("Average Article Count per Message")
plt.legend(title="Article Type")
plt.grid(True)
plt.tight_layout()
plt.show()
