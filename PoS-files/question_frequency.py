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

# Question detection function 
wh_words = {"what", "where", "who", "when", "why", "how", "which", "whose"}

def is_question(text):
    if pd.isna(text):
        return 0
    text = text.lower().strip()
    # Ends with question mark
    if text.endswith("?"):
        return 1
    # Starts with a wh-word
    first_word = re.split(r"\W+", text)[0]
    if first_word in wh_words:
        return 1
    return 0

df["is_question"] = df["text"].apply(is_question)

# Aggregate over rep_num
question_by_rep = df.groupby("rep_num")["is_question"].mean().reset_index()

# Plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=question_by_rep, x="rep_num", y="is_question", marker="o")
plt.title(f"{paper_name}: Average Question Frequency Over Repetitions")
plt.xlabel("Repetition Number")
plt.ylabel("Proportion of Describer Messages that Are Questions")
plt.grid(True)
plt.tight_layout()
plt.show()
